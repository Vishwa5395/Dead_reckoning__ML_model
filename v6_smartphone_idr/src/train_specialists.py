"""
train_specialists.py
--------------------
Independent Training Engines for Dual-Specialist PINO-DR:
  - Model A (Turning Specialist): Trained ONLY on Turning Family Dynamics
  - Model B (Longitudinal Specialist): Trained ONLY on Longitudinal Family Dynamics

Zero Gradient Sharing:
  Model A parameters NEVER see Model B data.
  Model B parameters NEVER see Model A data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
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

# Multi-threaded CPU execution
torch.set_num_threads(min(8, os.cpu_count() or 4))

from v6_smartphone_idr.src.models_specialists import (
    TurningBackboneModelA,
    LongitudinalBackboneModelB,
)

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "v6_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 1. Training Engine for Model A (Turning Specialist)
# ============================================================================

def train_specialist_A(
    max_epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 6,
):
    set_seed(42)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STARTING INDEPENDENT TRAINING: MODEL A (TURNING SPECIALIST)")
    print("=" * 80)

    npz_path = DATA_DIR / "dataset_specialists.npz"
    scalers_path = DATA_DIR / "scalers_specialists.pkl"
    d = np.load(npz_path)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)["A"]

    X_tr = torch.tensor(d["X_tr_A"], dtype=torch.float32)
    y_dv_tr = torch.tensor(d["y_dv_tr_A"], dtype=torch.float32)
    y_w_tr = torch.tensor(d["y_w_tr_A"], dtype=torch.float32)
    y_bw_tr = torch.tensor(d["y_bw_tr_A"], dtype=torch.float32)

    X_va = torch.tensor(d["X_va_A"], dtype=torch.float32)
    y_dv_va = torch.tensor(d["y_dv_va_A"], dtype=torch.float32)
    y_w_va = torch.tensor(d["y_w_va_A"], dtype=torch.float32)
    y_bw_va = torch.tensor(d["y_bw_va_A"], dtype=torch.float32)

    print(f"[Model A] Training windows:   {len(X_tr):,}")
    print(f"[Model A] Validation windows: {len(X_va):,}")

    ds_tr = TensorDataset(X_tr, y_w_tr, y_dv_tr, y_bw_tr)
    ds_va = TensorDataset(X_va, y_w_va, y_dv_va, y_bw_va)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model A] Device: {device}")

    model = TurningBackboneModelA().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model A] Total Trainable Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    s_yw = scalers["y_w"]
    s_ydv = scalers["y_dv"]
    w_scale = float(s_yw.scale_[0])

    best_val_w_mae = float("inf")
    best_epoch = 0
    history = []
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_l_head = 0.0
        n_batches = 0

        for bx, b_yw, b_ydv, b_ybw in loader_tr:
            bx = bx.to(device)
            b_yw = b_yw.to(device)
            b_ydv = b_ydv.to(device)
            b_ybw = b_ybw.to(device)

            w_pred, dv_pred, bw_pred, log_var_w = model(bx)

            # Turn-weighted Huber loss on zero-centered standardized yaw rate
            w_phys = b_yw * w_scale
            turn_mult = 1.0 + 3.0 * torch.clamp(torch.abs(w_phys) / 0.10, min=0.0, max=3.0)
            l_head = (F.huber_loss(w_pred, b_yw, reduction="none", delta=0.04) * turn_mult).mean()

            # Coupled speed change loss
            l_dv = F.huber_loss(dv_pred, b_ydv, delta=0.05)

            # Gyro bias loss
            l_bias = F.huber_loss(bw_pred, b_ybw, delta=0.04)

            # Heteroscedastic uncertainty loss
            err_w = (w_pred - b_yw) ** 2
            l_unc = (0.5 * torch.exp(-log_var_w) * err_w + 0.5 * log_var_w + 1.0).mean()

            # Multi-task loss emphasizing yaw mastery
            loss = 4.0 * l_head + 0.8 * l_dv + 0.15 * l_bias + 0.02 * l_unc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l_head += l_head.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / n_batches
        train_head_loss = total_l_head / n_batches

        # Validation: Physical yaw rate MAE (rad/s and deg/s)
        model.eval()
        val_w_errs = []
        val_dv_errs = []
        with torch.no_grad():
            for bx, b_yw, b_ydv, b_ybw in loader_va:
                bx = bx.to(device)
                w_pred, dv_pred, _, _ = model(bx)

                w_p_phys = s_yw.inverse_transform(w_pred.cpu().numpy())
                w_t_phys = s_yw.inverse_transform(b_yw.numpy())
                dv_p_phys = s_ydv.inverse_transform(dv_pred.cpu().numpy())
                dv_t_phys = s_ydv.inverse_transform(b_ydv.numpy())

                val_w_errs.append(np.abs(w_p_phys - w_t_phys).mean())
                val_dv_errs.append(np.abs(dv_p_phys - dv_t_phys).mean())

        val_w_mae = float(np.mean(val_w_errs))   # in rad/s
        val_dv_mae = float(np.mean(val_dv_errs)) # in m/s
        val_w_deg_s = val_w_mae * 180.0 / math.pi
        dt_epoch = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_head_loss": train_head_loss,
            "val_w_mae_rad_s": val_w_mae,
            "val_w_mae_deg_s": val_w_deg_s,
            "val_dv_mae_m_s": val_dv_mae,
            "epoch_sec": dt_epoch,
            "lr": float(scheduler.get_last_lr()[0]),
        })

        improved = val_w_mae < best_val_w_mae
        flag = "(*BEST*)" if improved else ""
        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{dt_epoch:.1f}s] | Loss: {train_loss:.4f} | Val Yaw MAE: {val_w_deg_s:.2f} deg/s ({val_w_mae:.4f} rad/s) | Val dv: {val_dv_mae:.4f} m/s {flag}")

        if improved:
            best_val_w_mae = val_w_mae
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_w_mae": val_w_mae,
                "val_w_deg_s": val_w_deg_s,
                "val_dv_mae": val_dv_mae,
            }, CKPT_DIR / "best_specialist_A.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Model A] Early stopping triggered at epoch {epoch} (Best Epoch: {best_epoch}, Best Val Yaw MAE: {best_val_w_mae*180/math.pi:.2f} deg/s)")
                break

    with open(RESULTS_DIR / "training_history_specialist_A.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_w_mae": best_val_w_mae, "history": history}, f, indent=2)
    print(f"[Model A] Best Checkpoint saved to {CKPT_DIR / 'best_specialist_A.pth'}")
    return best_epoch, best_val_w_mae


# ============================================================================
# 2. Training Engine for Model B (Longitudinal Specialist)
# ============================================================================

def train_specialist_B(
    max_epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 6,
):
    set_seed(42)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STARTING INDEPENDENT TRAINING: MODEL B (LONGITUDINAL SPECIALIST)")
    print("=" * 80)

    npz_path = DATA_DIR / "dataset_specialists.npz"
    scalers_path = DATA_DIR / "scalers_specialists.pkl"
    d = np.load(npz_path)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)["B"]

    X_tr = torch.tensor(d["X_tr_B"], dtype=torch.float32)
    y_dv_tr = torch.tensor(d["y_dv_tr_B"], dtype=torch.float32)
    y_w_tr = torch.tensor(d["y_w_tr_B"], dtype=torch.float32)
    y_z_tr = torch.tensor(d["y_z_tr_B"], dtype=torch.float32)
    y_ba_tr = torch.tensor(d["y_ba_tr_B"], dtype=torch.float32)

    X_va = torch.tensor(d["X_va_B"], dtype=torch.float32)
    y_dv_va = torch.tensor(d["y_dv_va_B"], dtype=torch.float32)
    y_w_va = torch.tensor(d["y_w_va_B"], dtype=torch.float32)
    y_z_va = torch.tensor(d["y_z_va_B"], dtype=torch.float32)
    y_ba_va = torch.tensor(d["y_ba_va_B"], dtype=torch.float32)

    print(f"[Model B] Training windows:   {len(X_tr):,}")
    print(f"[Model B] Validation windows: {len(X_va):,}")

    ds_tr = TensorDataset(X_tr, y_dv_tr, y_w_tr, y_z_tr, y_ba_tr)
    ds_va = TensorDataset(X_va, y_dv_va, y_w_va, y_z_va, y_ba_va)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model B] Device: {device}")

    model = LongitudinalBackboneModelB().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model B] Total Trainable Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    s_ydv = scalers["y_dv"]
    s_yw = scalers["y_w"]
    dv_scale = float(s_ydv.scale_[0])

    best_val_score = float("inf")
    best_epoch = 0
    history = []
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_l_dv = 0.0
        n_batches = 0

        for bx, b_ydv, b_yw, b_yz, b_yba in loader_tr:
            bx = bx.to(device)
            b_ydv = b_ydv.to(device)
            b_yw = b_yw.to(device)
            b_yz = b_yz.to(device)
            b_yba = b_yba.to(device)

            dv_pred, w_pred, z_logit, ba_pred, log_var_v = model(bx)

            # Acceleration-weighted Huber loss on zero-centered standardized delta-v
            dv_phys = b_ydv * dv_scale
            accel_mult = 1.0 + 3.0 * torch.clamp(torch.abs(dv_phys) / 0.15, min=0.0, max=3.0)
            l_dv = (F.huber_loss(dv_pred, b_ydv, reduction="none", delta=0.04) * accel_mult).mean()

            # ZUPT standstill classification loss
            l_zupt = F.binary_cross_entropy_with_logits(z_logit, b_yz)

            # Secondary coupled yaw rate
            l_w = F.huber_loss(w_pred, b_yw, delta=0.04)

            # Accel residual loss
            l_bias = F.huber_loss(ba_pred, b_yba, delta=0.05)

            # Heteroscedastic uncertainty loss
            err_v = (dv_pred - b_ydv) ** 2
            l_unc = (0.5 * torch.exp(-log_var_v) * err_v + 0.5 * log_var_v + 1.0).mean()

            # Multi-task loss emphasizing speed and standstill mastery
            loss = 3.5 * l_dv + 1.0 * l_zupt + 0.5 * l_w + 0.15 * l_bias + 0.02 * l_unc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l_dv += l_dv.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / n_batches
        train_dv_loss = total_l_dv / n_batches

        # Validation: Velocity MAE & ZUPT accuracy
        model.eval()
        val_dv_errs = []
        val_w_errs = []
        zupt_correct = 0
        zupt_total = 0

        with torch.no_grad():
            for bx, b_ydv, b_yw, b_yz, b_yba in loader_va:
                bx = bx.to(device)
                dv_pred, w_pred, z_logit, _, _ = model(bx)

                dv_p_phys = s_ydv.inverse_transform(dv_pred.cpu().numpy())
                dv_t_phys = s_ydv.inverse_transform(b_ydv.numpy())
                w_p_phys = s_yw.inverse_transform(w_pred.cpu().numpy())
                w_t_phys = s_yw.inverse_transform(b_yw.numpy())

                val_dv_errs.append(np.abs(dv_p_phys - dv_t_phys).mean())
                val_w_errs.append(np.abs(w_p_phys - w_t_phys).mean())

                p_stop = (torch.sigmoid(z_logit) > 0.5).cpu().numpy()
                zupt_correct += (p_stop == b_yz.numpy()).sum()
                zupt_total += len(b_yz)

        val_dv_mae = float(np.mean(val_dv_errs))  # in m/s
        val_w_mae = float(np.mean(val_w_errs))    # in rad/s
        val_zupt_acc = float(zupt_correct / zupt_total * 100)
        dt_epoch = time.time() - t0

        val_score = 0.85 * (val_dv_mae / 0.50) + 0.15 * (1.0 - val_zupt_acc / 100.0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dv_loss": train_dv_loss,
            "val_dv_mae_m_s": val_dv_mae,
            "val_w_mae_rad_s": val_w_mae,
            "val_zupt_acc": val_zupt_acc,
            "val_score": val_score,
            "epoch_sec": dt_epoch,
            "lr": float(scheduler.get_last_lr()[0]),
        })

        improved = val_score < best_val_score
        flag = "(*BEST*)" if improved else ""
        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{dt_epoch:.1f}s] | Loss: {train_loss:.4f} | Val Speed MAE: {val_dv_mae:.4f} m/s | ZUPT Acc: {val_zupt_acc:.1f}% | Val Score: {val_score:.4f} {flag}")

        if improved:
            best_val_score = val_score
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_score": val_score,
                "val_dv_mae": val_dv_mae,
                "val_zupt_acc": val_zupt_acc,
            }, CKPT_DIR / "best_specialist_B.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Model B] Early stopping triggered at epoch {epoch} (Best Epoch: {best_epoch}, Best Val Speed MAE: {val_dv_mae:.4f} m/s)")
                break

    with open(RESULTS_DIR / "training_history_specialist_B.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_score": best_val_score, "history": history}, f, indent=2)
    print(f"[Model B] Best Checkpoint saved to {CKPT_DIR / 'best_specialist_B.pth'}")
    return best_epoch, best_val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "BOTH"], default="BOTH")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    if args.model in ["A", "BOTH"]:
        train_specialist_A(max_epochs=args.epochs)
    if args.model in ["B", "BOTH"]:
        train_specialist_B(max_epochs=args.epochs)
