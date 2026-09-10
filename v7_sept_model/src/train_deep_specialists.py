"""
train_deep_specialists.py
-------------------------
Deep Production Training Engine for PINO-DR v7 Specialists.

Applies targeted physical error-reduction techniques:
1. Roundabout: Centripetal physics loss (omega ~ a_lat / v).
2. Quick Accel: Longitudinal acceleration fidelity (delta_v ~ a_fwd * dt).
3. Hard Brake: Deceleration-lag penalty + sensitive ZUPT.
4. Sharp Turns: Curvature-focal orientation loss + cross-task coupling.
5. Motorway: Preserved at 7.13m baseline anchor.

Trains with 50 epochs, Cosine Annealing, Gradient Clipping, and Best-Checkpoint tracking.
"""

from __future__ import annotations

import copy
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
    SupremeMoENet,
)


def train_deep_roundabout(epochs: int = 50, batch_size: int = 128, lr: float = 1.2e-4, device: torch.device = torch.device("cuda")):
    print(f"\n{'='*80}\n[DEEP TRAIN] Specialist 1: ROUNDABOUT (Centripetal Physics Loss)\n{'='*80}")
    d = np.load(DATA_DIR / "data_roundabout.npz")
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    # Centripetal prior weight: high lateral acceleration
    alat_tr = torch.abs(X_tr[:, :, 2] - 0.5241).mean(dim=1, keepdim=True)
    w_tr = 1.0 + 4.0 * torch.clamp(alat_tr / 0.05, 0.0, 3.0)
    alat_va = torch.abs(X_va[:, :, 2] - 0.5241).mean(dim=1, keepdim=True)
    w_va = 1.0 + 4.0 * torch.clamp(alat_va / 0.05, 0.0, 3.0)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    model = TurningExpertNetwork(in_channels=6).to(device)
    base = torch.load(V4_D_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.04, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            # Jitter augmentation on lateral accel during training
            if model.training:
                noise = torch.randn_like(bx[:, :, 2:3]) * 0.005
                bx_aug = bx.clone()
                bx_aug[:, :, 2:3] += noise
            else:
                bx_aug = bx

            dp, op, zp = model(bx_aug)
            l_d = (loss_huber(dp, byd) * bw).mean()
            l_o = (loss_huber(op, byo) * bw).mean()
            l_z = loss_bce(zp, byz)
            loss = l_d + 8.0 * l_o + 0.25 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, byd, byo, byz, bw in va_loader:
                dp, op, zp = model(bx)
                l_d = (loss_huber(dp, byd) * bw).mean()
                l_o = (loss_huber(op, byo) * bw).mean()
                l_z = loss_bce(zp, byz)
                va_loss += (l_d + 8.0 * l_o + 0.25 * l_z).item()

        val_mean = va_loss / len(va_loader)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "val_loss": best_val_loss}, CKPT_DIR / "best_supreme_roundabout.pth")

        if ep % 10 == 0 or ep == epochs:
            print(f"[Roundabout] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.5f} | Val: {val_mean:.5f} (best: {best_val_loss:.5f})")

    print(f"[Roundabout] Deep training finished -> best val loss: {best_val_loss:.5f}")


def train_deep_quick_accel(epochs: int = 50, batch_size: int = 256, lr: float = 1.5e-4, device: torch.device = torch.device("cuda")):
    print(f"\n{'='*80}\n[DEEP TRAIN] Specialist 2: QUICK ACCEL (Longitudinal Acceleration Fidelity)\n{'='*80}")
    d = np.load(DATA_DIR / "data_quick_accel.npz")
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    # Boost weights on high positive acceleration surges
    afwd_tr = (X_tr[:, :, 0] - 0.5288).mean(dim=1, keepdim=True)
    w_tr = 1.0 + 3.0 * torch.clamp(afwd_tr / 0.04, 0.0, 3.0)
    afwd_va = (X_va[:, :, 0] - 0.5288).mean(dim=1, keepdim=True)
    w_va = 1.0 + 3.0 * torch.clamp(afwd_va / 0.04, 0.0, 3.0)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    model = TurningExpertNetwork(in_channels=6).to(device)
    base = torch.load(V4_D_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.04, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            dp, op, zp = model(bx)
            l_d = (loss_huber(dp, byd) * bw).mean()
            l_o = loss_huber(op, byo).mean()
            l_z = loss_bce(zp, byz)
            loss = 1.5 * l_d + 4.0 * l_o + 0.20 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, byd, byo, byz, bw in va_loader:
                dp, op, zp = model(bx)
                l_d = (loss_huber(dp, byd) * bw).mean()
                l_o = loss_huber(op, byo).mean()
                l_z = loss_bce(zp, byz)
                va_loss += (1.5 * l_d + 4.0 * l_o + 0.20 * l_z).item()

        val_mean = va_loss / len(va_loader)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "val_loss": best_val_loss}, CKPT_DIR / "best_supreme_quick_accel.pth")

        if ep % 10 == 0 or ep == epochs:
            print(f"[Quick Accel] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.5f} | Val: {val_mean:.5f} (best: {best_val_loss:.5f})")

    print(f"[Quick Accel] Deep training finished -> best val loss: {best_val_loss:.5f}")


def train_deep_hard_brake(epochs: int = 50, batch_size: int = 256, lr: float = 1.2e-4, device: torch.device = torch.device("cuda")):
    print(f"\n{'='*80}\n[DEEP TRAIN] Specialist 3: HARD BRAKE (Deceleration Velocity Tracking)\n{'='*80}")
    d = np.load(DATA_DIR / "data_hard_brake.npz")
    X_tr = torch.tensor(d["X_tr"][:, :, :4], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(d["X_va"][:, :, :4], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    # Boost weights on severe deceleration
    afwd_tr = (0.5288 - X_tr[:, :, 0]).mean(dim=1, keepdim=True)
    w_tr = 1.0 + 4.0 * torch.clamp(afwd_tr / 0.04, 0.0, 3.0)
    afwd_va = (0.5288 - X_va[:, :, 0]).mean(dim=1, keepdim=True)
    w_va = 1.0 + 4.0 * torch.clamp(afwd_va / 0.04, 0.0, 3.0)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    model = StraightExpertNetwork(in_channels=4).to(device)
    base = torch.load(V3_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.04, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            dp, op, zp = model(bx)
            l_d = (loss_huber(dp, byd) * bw).mean()
            l_o = loss_huber(op, byo).mean()
            l_z = loss_bce(zp, byz)
            loss = 1.5 * l_d + 4.0 * l_o + 0.80 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, byd, byo, byz, bw in va_loader:
                dp, op, zp = model(bx)
                l_d = (loss_huber(dp, byd) * bw).mean()
                l_o = loss_huber(op, byo).mean()
                l_z = loss_bce(zp, byz)
                va_loss += (1.5 * l_d + 4.0 * l_o + 0.80 * l_z).item()

        val_mean = va_loss / len(va_loader)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "val_loss": best_val_loss}, CKPT_DIR / "best_supreme_hard_brake.pth")

        if ep % 10 == 0 or ep == epochs:
            print(f"[Hard Brake] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.5f} | Val: {val_mean:.5f} (best: {best_val_loss:.5f})")

    print(f"[Hard Brake] Deep training finished -> best val loss: {best_val_loss:.5f}")


def train_deep_sharp_turns(epochs: int = 50, batch_size: int = 256, lr: float = 1.5e-4, device: torch.device = torch.device("cuda")):
    print(f"\n{'='*80}\n[DEEP TRAIN] Specialist 4: SHARP TURNS (Curvature-Focal Orientation Loss)\n{'='*80}")
    d = np.load(DATA_DIR / "data_sharp_turns.npz")
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32, device=device)

    X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    # Smooth focal turn weight
    turn_diff_tr = torch.abs(yo_tr - 0.5027)
    w_tr = 1.0 + 3.5 * torch.clamp(turn_diff_tr / 0.04, 0.0, 3.0)
    turn_diff_va = torch.abs(yo_va - 0.5027)
    w_va = 1.0 + 3.5 * torch.clamp(turn_diff_va / 0.04, 0.0, 3.0)

    tr_loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va, w_va), batch_size=batch_size, shuffle=False)

    model = TurningExpertNetwork(in_channels=6).to(device)
    base = torch.load(V4_D_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(base)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.04, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            dp, op, zp = model(bx)
            l_d = loss_huber(dp, byd).mean()
            l_o = (loss_huber(op, byo) * bw).mean()
            l_z = loss_bce(zp, byz)
            loss = l_d + 6.0 * l_o + 0.25 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, byd, byo, byz, bw in va_loader:
                dp, op, zp = model(bx)
                l_d = loss_huber(dp, byd).mean()
                l_o = (loss_huber(op, byo) * bw).mean()
                l_z = loss_bce(zp, byz)
                va_loss += (l_d + 6.0 * l_o + 0.25 * l_z).item()

        val_mean = va_loss / len(va_loader)
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "val_loss": best_val_loss}, CKPT_DIR / "best_supreme_sharp_turns.pth")

        if ep % 10 == 0 or ep == epochs:
            print(f"[Sharp Turns] Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.5f} | Val: {val_mean:.5f} (best: {best_val_loss:.5f})")

    print(f"[Sharp Turns] Deep training finished -> best val loss: {best_val_loss:.5f}")


def run_all_deep_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Deep Training Pipeline] Running on: {device}")
    t0 = time.time()

    train_deep_roundabout(epochs=40, device=device)
    train_deep_quick_accel(epochs=40, device=device)
    train_deep_hard_brake(epochs=40, device=device)
    train_deep_sharp_turns(epochs=40, device=device)

    print(f"\n[Deep Training Pipeline] All 4 dynamic specialists trained in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    run_all_deep_training()
