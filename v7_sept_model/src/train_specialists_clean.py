"""
train_specialists_clean.py
--------------------------
Trains the 5 regime specialists using the verified SupremeMoENet architectures:
- Expert 0: StraightExpertNetwork(4ch) -> Motorway (preserved from v3 baseline: 7.13m)
- Expert 1: TurningExpertNetwork(6ch)  -> Roundabout (trained on data_roundabout.npz)
- Expert 2: TurningExpertNetwork(6ch)  -> Quick Accel (trained on data_quick_accel.npz)
- Expert 3: StraightExpertNetwork(4ch) -> Hard Brake (trained on data_hard_brake.npz)
- Expert 4: TurningExpertNetwork(6ch)  -> Sharp Turns (trained on data_sharp_turns.npz)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

V3_CKPT = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
V4_D_CKPT = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"

from v7_sept_model.src.moe_five_model import (
    StraightExpertNetwork,
    TurningExpertNetwork,
)


def train_specialist(
    scen_name: str,
    model: nn.Module,
    init_ckpt: Path,
    is_4ch: bool = False,
    epochs: int = 35,
    batch_size: int = 256,
    lr: float = 1.5e-4,
    alpha_ori: float = 6.0,
    beta_zupt: float = 0.25,
    focal_turn: bool = False,
    focal_brake: bool = False,
    device: torch.device = torch.device("cuda"),
):
    print(f"\n{'='*75}\n[TRAINING] Specialist: {scen_name.upper()} ({'4-channel' if is_4ch else '6-channel'})\n{'='*75}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"data_{scen_name}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data partition: {data_path}")

    d = np.load(data_path)
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    if is_4ch:
        X_tr = X_tr[:, :, :4]
        X_va = X_va[:, :, :4]

    # Sample weights
    if focal_turn:
        diff_tr = torch.abs(yo_tr - 0.5027)
        diff_va = torch.abs(yo_va - 0.5027)
        w_tr = 1.0 + 5.0 * torch.clamp(diff_tr / 0.05, 0.0, 3.0)
        w_va = 1.0 + 5.0 * torch.clamp(diff_va / 0.05, 0.0, 3.0)
    elif focal_brake:
        # Penalize failing to brake when a_fwd is strongly negative
        afwd_tr = X_tr[:, :, 0].mean(dim=1, keepdim=True)
        afwd_va = X_va[:, :, 0].mean(dim=1, keepdim=True)
        w_tr = 1.0 + 3.0 * torch.clamp((0.5288 - afwd_tr) / 0.10, 0.0, 3.0)
        w_va = 1.0 + 3.0 * torch.clamp((0.5288 - afwd_va) / 0.10, 0.0, 3.0)
    else:
        w_tr = torch.ones_like(yd_tr)
        w_va = torch.ones_like(yd_va)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    print(f"[{scen_name}] Train windows: {len(X_tr):,} | Val windows: {len(X_va):,}")

    # Transfer base weights
    if init_ckpt.exists():
        print(f"[{scen_name}] Transferring base weights from: {init_ckpt.name}")
        base = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = base.get("model_state_dict", base)
        cur = model.state_dict()
        matched = {k: v for k, v in state.items() if k in cur and v.shape == cur[k].shape}
        cur.update(matched)
        model.load_state_dict(cur)
        print(f"[{scen_name}] Transferred {len(matched)} / {len(cur)} parameter tensors.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.05, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    save_path = CKPT_DIR / f"best_supreme_{scen_name}.pth"

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            dp, op, zp = model(bx)
            l_d = (loss_huber(dp, byd) * bw).mean()
            l_o = (loss_huber(op, byo) * bw).mean()
            l_z = loss_bce(zp, byz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, byd, byo, byz, bw in va_loader:
                dp, op, zp = model(bx)
                l_d = (loss_huber(dp, byd) * bw).mean()
                l_o = (loss_huber(op, byo) * bw).mean()
                l_z = loss_bce(zp, byz)
                loss = l_d + alpha_ori * l_o + beta_zupt * l_z
                va_loss += loss.item()

        val_mean = va_loss / max(len(va_loader), 1)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "val_loss": best_val_loss,
                "epoch": ep,
                "scen": scen_name,
            }, save_path)

        if ep % 5 == 0 or ep == epochs:
            print(f"[{scen_name}] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.5f} | Val: {val_mean:.5f} (best: {best_val_loss:.5f})")

    print(f"[{scen_name}] Training complete -> saved to {save_path.name} (best val loss: {best_val_loss:.5f})")


def train_all_specialists():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Specialists] Device: {device}")

    # 1. Motorway: Keep unchanged or lightly fine-tune
    # User instruction: "keep training until we get better results except for motorway"
    # We leave Expert 0 anchored to v3 weights!
    print("[Specialists] Preserving Expert 0 (Motorway) at its proven 7.13m / 7.27m baseline anchor.")

    # 2. Roundabout (Expert 1)
    m_rb = TurningExpertNetwork(in_channels=6)
    train_specialist(
        "roundabout", m_rb, V4_D_CKPT,
        is_4ch=False, epochs=30, lr=1.5e-4, alpha_ori=10.0, beta_zupt=0.25,
        focal_turn=True, device=device
    )

    # 3. Quick Accel (Expert 2)
    m_qa = TurningExpertNetwork(in_channels=6)
    train_specialist(
        "quick_accel", m_qa, V4_D_CKPT,
        is_4ch=False, epochs=30, lr=1.8e-4, alpha_ori=6.0, beta_zupt=0.20,
        device=device
    )

    # 4. Hard Brake (Expert 3) - StraightExpertNetwork 4ch
    m_hb = StraightExpertNetwork(in_channels=4)
    train_specialist(
        "hard_brake", m_hb, V3_CKPT,
        is_4ch=True, epochs=30, lr=1.5e-4, alpha_ori=6.0, beta_zupt=0.80,
        focal_brake=True, device=device
    )

    # 5. Sharp Turns (Expert 4) - TurningExpertNetwork 6ch
    m_st = TurningExpertNetwork(in_channels=6)
    train_specialist(
        "sharp_turns", m_st, V4_D_CKPT,
        is_4ch=False, epochs=35, lr=2.0e-4, alpha_ori=8.0, beta_zupt=0.25,
        focal_turn=True, device=device
    )

    print("\n[Specialists] All domain specialists trained successfully!")


if __name__ == "__main__":
    train_all_specialists()
