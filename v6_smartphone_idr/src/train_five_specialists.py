"""
train_five_specialists.py
-------------------------
Training Pipeline for Five Genuinely Independent Specialist Models:
  - Stage 1: Independent training of M1 (Velocity), M2 (Yaw), M3 (Accel Bias), M4 (Gyro Bias)
  - Stage 2: Empirical residual extraction & M5 (Uncertainty Adapter) Gaussian NLL training

Each model trains on its own quantity-specific window, loss function, and scheduler.
Zero shared weights or gradient paths.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v6_smartphone_idr.src.models_five_specialists import (
    VelocitySpecialistM1,
    YawSpecialistM2,
    AccelBiasSpecialistM3,
    GyroBiasSpecialistM4,
    UncertaintyAdapterM5,
)

DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Training Engine for M1: Velocity ────────────────────────────────────────

def train_specialist_M1(X_tr, y_tr, X_va, y_va, epochs=15, batch_size=512, lr=1e-3):
    print("\n" + "=" * 70)
    print("  Training Specialist M1: Forward Velocity & Speed Propagation")
    print("=" * 70)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VelocitySpecialistM1().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    huber = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINTS_DIR / "best_specialist_M1.pth"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_v_loss, total_dv_loss = 0.0, 0.0, 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()

            pred_v, pred_dv = model(bx)
            loss_v = huber(pred_v, by[:, 0:1])
            loss_dv = huber(pred_dv, by[:, 1:2])

            loss = loss_v + 1.5 * loss_dv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += loss_v.item()
            total_dv_loss += loss_dv.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_v_mae = 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred_v, pred_dv = model(bx)
                loss_v = huber(pred_v, by[:, 0:1])
                loss_dv = huber(pred_dv, by[:, 1:2])
                val_loss += (loss_v + 1.5 * loss_dv).item()
                val_v_mae += F.l1_loss(pred_v, by[:, 0:1]).item()
                val_batches += 1

        val_loss /= val_batches
        val_v_mae /= val_batches
        dt = time.time() - t0

        print(f"  Epoch {ep:02d}/{epochs:02d} [{dt:.1f}s] - Train Loss: {total_loss/n_batches:.4f} "
              f"- Val Loss: {val_loss:.4f} - Val v MAE: {val_v_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  [M1] Saved Best Checkpoint -> {ckpt_path} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ─── Training Engine for M2: Yaw Rate ────────────────────────────────────────

def train_specialist_M2(X_tr, y_tr, X_va, y_va, epochs=15, batch_size=512, lr=1e-3):
    print("\n" + "=" * 70)
    print("  Training Specialist M2: Yaw Rate & Heading Dynamics")
    print("=" * 70)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = YawSpecialistM2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    huber = nn.HuberLoss(delta=0.5)

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINTS_DIR / "best_specialist_M2.pth"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()

            pred_w, pred_dpsi = model(bx)
            # Turn weighting for dynamic turns
            turn_weight = 1.0 + 2.0 * torch.clamp(torch.abs(by[:, 0:1]), 0.0, 3.0)

            loss_w = (huber(pred_w, by[:, 0:1]) * turn_weight).mean()
            loss_dpsi = huber(pred_dpsi, by[:, 1:2])
            loss = loss_w + loss_dpsi

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_w_mae = 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred_w, pred_dpsi = model(bx)
                val_loss += (huber(pred_w, by[:, 0:1]) + huber(pred_dpsi, by[:, 1:2])).item()
                val_w_mae += F.l1_loss(pred_w, by[:, 0:1]).item()
                val_batches += 1

        val_loss /= val_batches
        val_w_mae /= val_batches
        dt = time.time() - t0

        print(f"  Epoch {ep:02d}/{epochs:02d} [{dt:.1f}s] - Train Loss: {total_loss/n_batches:.4f} "
              f"- Val Loss: {val_loss:.4f} - Val w MAE: {val_w_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  [M2] Saved Best Checkpoint -> {ckpt_path} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ─── Training Engine for M3: Accel Bias ───────────────────────────────────────

def train_specialist_M3(X_tr, y_tr, X_va, y_va, epochs=12, batch_size=512, lr=1e-3):
    print("\n" + "=" * 70)
    print("  Training Specialist M3: Accelerometer Bias (Slow Drift)")
    print("=" * 70)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AccelBiasSpecialistM3().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINTS_DIR / "best_specialist_M3.pth"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()

            pred_ba = model(bx)
            loss = loss_fn(pred_ba, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += loss_fn(model(bx), by).item()
                val_batches += 1

        val_loss /= val_batches
        dt = time.time() - t0

        print(f"  Epoch {ep:02d}/{epochs:02d} [{dt:.1f}s] - Train Loss: {total_loss/n_batches:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  [M3] Saved Best Checkpoint -> {ckpt_path} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ─── Training Engine for M4: Gyro Bias ────────────────────────────────────────

def train_specialist_M4(X_tr, y_tr, X_va, y_va, epochs=12, batch_size=512, lr=1e-3):
    print("\n" + "=" * 70)
    print("  Training Specialist M4: Gyroscope Bias (Slow Drift)")
    print("=" * 70)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GyroBiasSpecialistM4().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINTS_DIR / "best_specialist_M4.pth"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()

            pred_bg = model(bx)
            loss = loss_fn(pred_bg, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += loss_fn(model(bx), by).item()
                val_batches += 1

        val_loss /= val_batches
        dt = time.time() - t0

        print(f"  Epoch {ep:02d}/{epochs:02d} [{dt:.1f}s] - Train Loss: {total_loss/n_batches:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  [M4] Saved Best Checkpoint -> {ckpt_path} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ─── Training Engine for M5: Uncertainty Adapter (Gaussian NLL) ──────────────

def train_adapter_M5(m1, m2, m3, m4, X_tr_dict, X_va_dict, epochs=10, batch_size=512, lr=1e-3):
    print("\n" + "=" * 70)
    print("  Training Specialist M5: Dynamic Uncertainty / Covariance Adapter")
    print("=" * 70)

    # 1. Compute empirical residuals of M1..M4 on training and validation splits
    def compute_residuals(m1, m2, m3, m4, x_dict):
        m1.eval()
        m2.eval()
        m3.eval()
        m4.eval()
        with torch.no_grad():
            bx1 = torch.tensor(x_dict["X_M1"], dtype=torch.float32).to(DEVICE)
            by1 = torch.tensor(x_dict["y_M1"], dtype=torch.float32).to(DEVICE)
            pred_v, _ = m1(bx1)
            err_v = torch.abs(pred_v - by1[:, 0:1])

            bx2 = torch.tensor(x_dict["X_M2"], dtype=torch.float32).to(DEVICE)
            by2 = torch.tensor(x_dict["y_M2"], dtype=torch.float32).to(DEVICE)
            pred_w, _ = m2(bx2)
            err_w = torch.abs(pred_w - by2[:, 0:1])

            bx3 = torch.tensor(x_dict["X_M3"], dtype=torch.float32).to(DEVICE)
            by3 = torch.tensor(x_dict["y_M3"], dtype=torch.float32).to(DEVICE)
            pred_ba = m3(bx3)
            err_ba = torch.abs(pred_ba - by3)

            bx4 = torch.tensor(x_dict["X_M4"], dtype=torch.float32).to(DEVICE)
            by4 = torch.tensor(x_dict["y_M4"], dtype=torch.float32).to(DEVICE)
            pred_bg = m4(bx4)
            err_bg = torch.abs(pred_bg - by4)

            residuals = torch.cat([err_v, err_w, err_ba, err_bg], dim=-1)
            return residuals.cpu().numpy()

    residuals_tr = compute_residuals(m1, m2, m3, m4, X_tr_dict)
    residuals_va = compute_residuals(m1, m2, m3, m4, X_va_dict)

    print(f"  Empirical Residuals (Train): Mean [v={residuals_tr[:, 0].mean():.3f}, w={residuals_tr[:, 1].mean():.3f}, "
          f"ba={residuals_tr[:, 2].mean():.3f}, bg={residuals_tr[:, 3].mean():.3f}]")

    train_ds = TensorDataset(torch.tensor(X_tr_dict["X_M5"], dtype=torch.float32), torch.tensor(residuals_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va_dict["X_M5"], dtype=torch.float32), torch.tensor(residuals_va, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UncertaintyAdapterM5().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def gaussian_nll(pred_sigma, target_err):
        # NLL = 0.5 * (log(sigma^2) + err^2 / sigma^2)
        var = pred_sigma**2
        return 0.5 * (torch.log(var) + (target_err**2) / var).mean()

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINTS_DIR / "best_specialist_M5.pth"

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()

            pred_sigma = model(bx)
            loss = gaussian_nll(pred_sigma, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += gaussian_nll(model(bx), by).item()
                val_batches += 1

        val_loss /= val_batches
        dt = time.time() - t0

        print(f"  Epoch {ep:02d}/{epochs:02d} [{dt:.1f}s] - Train NLL: {total_loss/n_batches:.4f} - Val NLL: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"  [M5] Saved Best Checkpoint -> {ckpt_path} (Val NLL: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def main():
    print("=" * 80)
    print("  PINO-DR: Independent 5-Specialist Training Pipeline")
    print(f"  Target Device: {DEVICE}")
    print("=" * 80)

    dataset_path = DATA_DIR / "dataset_five_specialists.npz"
    if not dataset_path.exists():
        print(f"[Error] Dataset not found at {dataset_path}. Run preprocess_five_specialists.py first!")
        sys.exit(1)

    print(f"[Train] Loading preprocessed multi-scale dataset from {dataset_path}...")
    d = np.load(dataset_path)

    # 1. Train M1
    m1 = train_specialist_M1(d["X_tr_M1"], d["y_tr_M1"], d["X_va_M1"], d["y_va_M1"], epochs=15)

    # 2. Train M2
    m2 = train_specialist_M2(d["X_tr_M2"], d["y_tr_M2"], d["X_va_M2"], d["y_va_M2"], epochs=15)

    # 3. Train M3
    m3 = train_specialist_M3(d["X_tr_M3"], d["y_tr_M3"], d["X_va_M3"], d["y_va_M3"], epochs=12)

    # 4. Train M4
    m4 = train_specialist_M4(d["X_tr_M4"], d["y_tr_M4"], d["X_va_M4"], d["y_va_M4"], epochs=12)

    # 5. Train M5 (Uncertainty Adapter using residuals of M1..M4)
    X_tr_dict = {"X_M1": d["X_tr_M1"], "y_M1": d["y_tr_M1"], "X_M2": d["X_tr_M2"], "y_M2": d["y_tr_M2"],
                 "X_M3": d["X_tr_M3"], "y_M3": d["y_tr_M3"], "X_M4": d["X_tr_M4"], "y_M4": d["y_tr_M4"], "X_M5": d["X_tr_M5"]}
    X_va_dict = {"X_M1": d["X_va_M1"], "y_M1": d["y_va_M1"], "X_M2": d["X_va_M2"], "y_M2": d["y_va_M2"],
                 "X_M3": d["X_va_M3"], "y_M3": d["y_va_M3"], "X_M4": d["X_va_M4"], "y_M4": d["y_va_M4"], "X_M5": d["X_va_M5"]}

    m5 = train_adapter_M5(m1, m2, m3, m4, X_tr_dict, X_va_dict, epochs=10)

    print("\n" + "=" * 80)
    print("  ALL FIVE SPECIALISTS SUCCESSFULLY TRAINED & CHECKPOINTED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
