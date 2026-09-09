"""
train_five_specialists.py
-------------------------
Parallel Fine-Tuning Engine for the 5 Supreme Specialists.
Trains each specialist on its dedicated regime pool with custom physical loss objectives.
"""

from __future__ import annotations

import argparse
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

from v7_sept_model.src.models_supreme_five import (
    MotorwaySpecialist,
    RoundaboutSpecialist,
    QuickAccelSpecialist,
    HardBrakeSpecialist,
    SharpTurnsSpecialist,
)

V3_CKPT = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
V4_D_CKPT = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"


def train_single_specialist(
    scen_name: str,
    model: nn.Module,
    init_ckpt_path: Path,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 2e-4,
    alpha_ori: float = 6.0,
    beta_zupt: float = 0.25,
    use_focal_turn: bool = False,
    device: torch.device = torch.device("cuda"),
):
    print(f"\n{'='*75}\n[TRAIN] Supreme Specialist: {scen_name.upper()}\n{'='*75}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"data_{scen_name}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Partitioned data {data_path} not found.")

    data = np.load(data_path)
    X_tr = torch.tensor(data["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(data["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(data["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(data["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(data["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(data["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(data["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(data["yz_va"], dtype=torch.float32, device=device)

    # 4ch slicing for Motorway (uses 4ch)
    if scen_name == "motorway":
        X_tr = X_tr[:, :, :4]
        X_va = X_va[:, :, :4]

    # Focal weights
    if use_focal_turn:
        diff_tr = torch.abs(yo_tr - 0.504565)
        diff_va = torch.abs(yo_va - 0.504565)
        w_tr = (1.0 + 12.0 * torch.clamp(diff_tr / 0.10, 0.0, 3.5) ** 2).to(device)
        w_va = (1.0 + 12.0 * torch.clamp(diff_va / 0.10, 0.0, 3.5) ** 2).to(device)
    else:
        w_tr = torch.ones_like(yd_tr)
        w_va = torch.ones_like(yd_va)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    print(f"[{scen_name}] Train windows: {len(X_tr):,} | Val windows: {len(X_va):,}")

    # Transfer base weights
    if init_ckpt_path.exists():
        print(f"[{scen_name}] Transferring base weights from: {init_ckpt_path.name}")
        base_state = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        m_dict = model.state_dict()
        filtered = {k: v for k, v in base_state["model_state_dict"].items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(filtered)
        model.load_state_dict(m_dict)
        print(f"[{scen_name}] Transferred {len(filtered)} / {len(m_dict)} parameter tensors.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.08, reduction="none")
    loss_zupt = nn.BCEWithLogitsLoss()

    best_va = float("inf")
    best_state = None
    best_ep = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for X, yd, yo, yz, w in tr_loader:
            optimizer.zero_grad()
            d_p, o_p, z_p = model(X)
            l_d = loss_huber(d_p, yd).mean()
            l_o = (loss_huber(o_p, yo) * w).mean()
            l_z = loss_zupt(z_p, yz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        va_loss = 0.0
        nb = 0
        with torch.no_grad():
            for X, yd, yo, yz, w in va_loader:
                d_p, o_p, z_p = model(X)
                l_d = loss_huber(d_p, yd).mean()
                l_o = (loss_huber(o_p, yo) * w).mean()
                l_z = loss_zupt(z_p, yz)
                loss = l_d + alpha_ori * l_o + beta_zupt * l_z
                va_loss += loss.item()
                nb += 1
        va_loss /= max(nb, 1)

        if va_loss < best_va:
            best_va = va_loss
            best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs:
            print(f"[{scen_name}] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.4f} | Val: {va_loss:.4f} (best ep {best_ep}: {best_va:.4f})")

    out_ckpt = CKPT_DIR / f"best_supreme_{scen_name}.pth"
    torch.save({"model_state_dict": best_state, "val_loss": best_va, "epoch": best_ep}, out_ckpt)
    print(f"[{scen_name}] Saved best checkpoint -> {out_ckpt.name} (elapsed: {time.time()-t0:.1f}s)")


def train_all_five(epochs: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v7][5-Supreme] Training on: {device}")

    # 1. Motorway
    m1 = MotorwaySpecialist(in_channels=4)
    train_single_specialist(
        "motorway", m1, V3_CKPT,
        epochs=epochs, lr=1.5e-4, alpha_ori=8.0, beta_zupt=0.25, use_focal_turn=False, device=device
    )

    # 2. Hard Brake
    m2 = HardBrakeSpecialist(in_channels=6)
    train_single_specialist(
        "hard_brake", m2, V4_D_CKPT,
        epochs=epochs, lr=2.0e-4, alpha_ori=6.0, beta_zupt=0.50, use_focal_turn=False, device=device
    )

    # 3. Quick Accel
    m3 = QuickAccelSpecialist(in_channels=6)
    train_single_specialist(
        "quick_accel", m3, V4_D_CKPT,
        epochs=epochs, lr=2.0e-4, alpha_ori=6.0, beta_zupt=0.20, use_focal_turn=False, device=device
    )

    # 4. Sharp Turns
    m4 = SharpTurnsSpecialist(in_channels=6)
    train_single_specialist(
        "sharp_turns", m4, V4_D_CKPT,
        epochs=epochs, lr=2.5e-4, alpha_ori=10.0, beta_zupt=0.25, use_focal_turn=True, device=device
    )

    # 5. Roundabout
    m5 = RoundaboutSpecialist(in_channels=6)
    train_single_specialist(
        "roundabout", m5, V4_D_CKPT,
        epochs=epochs, lr=2.5e-4, alpha_ori=12.0, beta_zupt=0.25, use_focal_turn=True, device=device
    )

    print("\n" + "="*80 + "\n[v7][5-Supreme] All 5 Supreme Specialists successfully trained & saved!\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train_all_five(epochs=args.epochs)
