"""
train_v7.py
-----------
Fine-Tuning Engine for PINO-DR v7 Dual-Specialist System (v7_sept_model).

Specialists:
1. StraightSpecialistS1:
   - Initialized from v3 PINO-DR checkpoint (`v3_pino_dr/checkpoints/best_model.pth`).
   - Plain Huber loss across displacement, yaw rate, and ZUPT.
   - Early stopping strictly on low-yaw validation slice (|yaw_rate| < tau_60).
   - Saved to checkpoints/best_specialist_S1.pth (+ ONNX + TorchScript).

2. TurningSpecialistS2:
   - Initialized from v4 Ablation-D checkpoint (`v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth`).
   - 6-channel input including yaw-acceleration and centripetal-residual.
   - Plain Huber loss across displacement, yaw rate, and ZUPT.
   - Early stopping strictly on high-yaw validation slice (|yaw_rate| >= tau_60).
   - Saved to checkpoints/best_specialist_S2.pth (+ ONNX + TorchScript).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from v7_sept_model.src.models_v7 import (
    StraightSpecialistS1,
    TurningSpecialistS2,
    export_onnx_specialist,
    export_torchscript_specialist,
)

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

V3_CKPT_PATH = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
V4_D_CKPT_PATH = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_partitioned_data():
    npz_path = DATA_DIR / "dataset_splits_v7.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Partitioned data not found: {npz_path}. Run split_data_v7.py first.")
    return np.load(npz_path)


# ─── Training Engine for S1 (Straight Specialist) ────────────────────────────

def train_specialist_s1(
    max_epochs: int = 50,
    patience: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.1,
    alpha_ori: float = 1.0,
    beta_zupt: float = 0.25,
    grad_clip: float = 1.0,
):
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING: SPECIALIST S1 (STRAIGHT DRIVING)")
    print("=" * 80)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S1] Training device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_partitioned_data()

    # Train on anchor replay pool (85% target + 15% cross-regime anchors)
    X_tr = torch.tensor(data["X_tr_s1_replay"], dtype=torch.float32, device=device)
    y_d_tr = torch.tensor(data["y_d_tr_s1_replay"], dtype=torch.float32, device=device)
    y_o_tr = torch.tensor(data["y_o_tr_s1_replay"], dtype=torch.float32, device=device)
    y_z_tr = torch.tensor(data["y_z_tr_s1_replay"], dtype=torch.float32, device=device)

    X_va = torch.tensor(data["X_va_s1"], dtype=torch.float32, device=device)
    y_d_va = torch.tensor(data["y_d_va_s1"], dtype=torch.float32, device=device)
    y_o_va = torch.tensor(data["y_o_va_s1"], dtype=torch.float32, device=device)
    y_z_va = torch.tensor(data["y_z_va_s1"], dtype=torch.float32, device=device)

    tr_loader = DataLoader(TensorDataset(X_tr, y_d_tr, y_o_tr, y_z_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_d_va, y_o_va, y_z_va), batch_size=batch_size, shuffle=False)

    print(f"[S1] Training windows (Anchor Replay Pool): {len(X_tr):,}")
    print(f"[S1] Held-out val windows (< tau_60):       {len(X_va):,}")

    # Model instantiation & checkpoint initialization
    model = StraightSpecialistS1(
        in_channels=4, conv_channels=32, gru_hidden=32, num_gru_layers=1, dropout=0.20
    ).to(device)

    if not V3_CKPT_PATH.exists():
        raise FileNotFoundError(f"v3 checkpoint not found: {V3_CKPT_PATH}")
    print(f"[S1] Loading pretrained weights from v3 PINO-DR: {V3_CKPT_PATH}")
    ckpt_v3 = torch.load(V3_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_v3["model_state_dict"])
    print("[S1] Weights successfully transferred from v3 PINO-DR.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    loss_fn_huber = nn.HuberLoss(delta=huber_delta)
    loss_fn_zupt = nn.BCEWithLogitsLoss()

    # Pretrained baseline evaluation (Epoch 0)
    model.eval()
    e0_loss, e0_disp_mae, e0_ori_mae = 0.0, 0.0, 0.0
    va_b = 0
    with torch.no_grad():
        for X, yd, yo, yz in va_loader:
            d_pred, o_pred, z_logit = model(X)
            l_d = loss_fn_huber(d_pred, yd)
            l_o = loss_fn_huber(o_pred, yo)
            l_z = loss_fn_zupt(z_logit, yz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z
            e0_loss += loss.item()
            e0_disp_mae += nn.functional.l1_loss(d_pred, yd).item()
            e0_ori_mae += nn.functional.l1_loss(o_pred, yo).item()
            va_b += 1
    e0_loss /= max(va_b, 1)
    e0_disp_mae /= max(va_b, 1)
    e0_ori_mae /= max(va_b, 1)
    print(f"[S1 Epoch 00/Pretrained] Val Loss: {e0_loss:.6f} | Disp MAE: {e0_disp_mae:.5f} | Ori MAE: {e0_ori_mae:.5f}")

    best_val_loss = e0_loss
    best_epoch = 0
    patience_counter = 0
    history = [{
        "epoch": 0,
        "train_loss": float("nan"),
        "val_loss": e0_loss,
        "val_disp_mae": e0_disp_mae,
        "val_ori_mae": e0_ori_mae,
        "lr": learning_rate,
    }]

    # Save initial pretrained state as fallback best
    best_state = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": e0_loss,
        "val_disp_mae": e0_disp_mae,
        "val_ori_mae": e0_ori_mae,
        "config": {
            "model": "StraightSpecialistS1",
            "in_channels": 4,
            "conv_channels": 32,
            "gru_hidden": 32,
            "num_gru_layers": 1,
            "dropout": 0.20,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    torch.save(best_state, CKPT_DIR / "best_specialist_S1.pth")

    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        total_loss, total_ld, total_lo, total_lz = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for X, yd, yo, yz in tr_loader:
            optimizer.zero_grad()
            d_pred, o_pred, z_logit = model(X)

            l_d = loss_fn_huber(d_pred, yd)
            l_o = loss_fn_huber(o_pred, yo)
            l_z = loss_fn_zupt(z_logit, yz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ld += l_d.item()
            total_lo += l_o.item()
            total_lz += l_z.item()
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)
        tr_loss = total_loss / nb

        # ── Validation (Strictly on Held-Out Low-Yaw Slice) ──
        model.eval()
        va_loss, va_disp_mae, va_ori_mae = 0.0, 0.0, 0.0
        va_batches = 0
        with torch.no_grad():
            for X, yd, yo, yz in va_loader:
                d_pred, o_pred, z_logit = model(X)
                l_d = loss_fn_huber(d_pred, yd)
                l_o = loss_fn_huber(o_pred, yo)
                l_z = loss_fn_zupt(z_logit, yz)
                loss = l_d + alpha_ori * l_o + beta_zupt * l_z

                va_loss += loss.item()
                va_disp_mae += nn.functional.l1_loss(d_pred, yd).item()
                va_ori_mae += nn.functional.l1_loss(o_pred, yo).item()
                va_batches += 1

        nva = max(va_batches, 1)
        va_loss /= nva
        va_disp_mae /= nva
        va_ori_mae /= nva

        lr_curr = optimizer.param_groups[0]["lr"]
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_disp_mae": va_disp_mae,
            "val_ori_mae": va_ori_mae,
            "lr": lr_curr,
        }
        history.append(rec)

        improved = va_loss < best_val_loss
        star = " *" if improved else ""
        print(
            f"[S1 Epoch {epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {tr_loss:.6f} | "
            f"Val Loss (Low-Yaw): {va_loss:.6f} | "
            f"Disp MAE: {va_disp_mae:.5f} | "
            f"Ori MAE: {va_ori_mae:.5f} | "
            f"LR: {lr_curr:.6f}{star}"
        )

        if improved:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va_loss,
                "val_disp_mae": va_disp_mae,
                "val_ori_mae": va_ori_mae,
                "config": {
                    "model": "StraightSpecialistS1",
                    "in_channels": 4,
                    "conv_channels": 32,
                    "gru_hidden": 32,
                    "num_gru_layers": 1,
                    "dropout": 0.20,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            }
            torch.save(best_state, CKPT_DIR / "best_specialist_S1.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[S1] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    elapsed = time.time() - t_start
    print(f"[S1] Finished in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")

    # Load best weights for export
    ckpt = torch.load(CKPT_DIR / "best_specialist_S1.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export_onnx_specialist(model, str(CKPT_DIR / "best_specialist_S1.onnx"), in_channels=4, device=device)
    export_torchscript_specialist(model, str(CKPT_DIR / "best_specialist_S1_torchscript.pt"), in_channels=4, device=device)

    with open(RESULTS_DIR / "training_history_S1.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_val_loss


# ─── Training Engine for S2 (Turning Specialist) ─────────────────────────────

def train_specialist_s2(
    max_epochs: int = 50,
    patience: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.1,
    alpha_ori: float = 1.0,
    beta_zupt: float = 0.25,
    grad_clip: float = 1.0,
):
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING: SPECIALIST S2 (TURNING DYNAMICS)")
    print("=" * 80)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S2] Training device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_partitioned_data()

    # Train on anchor replay pool (85% target + 15% cross-regime anchors)
    X_tr = torch.tensor(data["X_tr_s2_replay"], dtype=torch.float32, device=device)
    y_d_tr = torch.tensor(data["y_d_tr_s2_replay"], dtype=torch.float32, device=device)
    y_o_tr = torch.tensor(data["y_o_tr_s2_replay"], dtype=torch.float32, device=device)
    y_z_tr = torch.tensor(data["y_z_tr_s2_replay"], dtype=torch.float32, device=device)

    X_va = torch.tensor(data["X_va_s2"], dtype=torch.float32, device=device)
    y_d_va = torch.tensor(data["y_d_va_s2"], dtype=torch.float32, device=device)
    y_o_va = torch.tensor(data["y_o_va_s2"], dtype=torch.float32, device=device)
    y_z_va = torch.tensor(data["y_z_va_s2"], dtype=torch.float32, device=device)

    tr_loader = DataLoader(TensorDataset(X_tr, y_d_tr, y_o_tr, y_z_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_d_va, y_o_va, y_z_va), batch_size=batch_size, shuffle=False)

    print(f"[S2] Training windows (Anchor Replay Pool): {len(X_tr):,}")
    print(f"[S2] Held-out val windows (>= tau_60):      {len(X_va):,}")

    # Model instantiation & checkpoint initialization
    model = TurningSpecialistS2(
        in_channels=6,
        conv_channels=32,
        gru_hidden=32,
        num_gru_layers=1,
        dropout=0.20,
        use_multihead_attention=True,
        use_cross_task_coupling=True,
    ).to(device)

    if not V4_D_CKPT_PATH.exists():
        raise FileNotFoundError(f"v4 Ablation-D checkpoint not found: {V4_D_CKPT_PATH}")
    print(f"[S2] Loading pretrained weights from v4 Ablation-D: {V4_D_CKPT_PATH}")
    ckpt_v4_d = torch.load(V4_D_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_v4_d["model_state_dict"])
    print("[S2] Weights successfully transferred from v4 Ablation-D.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    loss_fn_huber = nn.HuberLoss(delta=huber_delta)
    loss_fn_zupt = nn.BCEWithLogitsLoss()

    # Pretrained baseline evaluation (Epoch 0)
    model.eval()
    e0_loss, e0_disp_mae, e0_ori_mae = 0.0, 0.0, 0.0
    va_b = 0
    with torch.no_grad():
        for X, yd, yo, yz in va_loader:
            d_pred, o_pred, z_logit = model(X)
            l_d = loss_fn_huber(d_pred, yd)
            l_o = loss_fn_huber(o_pred, yo)
            l_z = loss_fn_zupt(z_logit, yz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z
            e0_loss += loss.item()
            e0_disp_mae += nn.functional.l1_loss(d_pred, yd).item()
            e0_ori_mae += nn.functional.l1_loss(o_pred, yo).item()
            va_b += 1
    e0_loss /= max(va_b, 1)
    e0_disp_mae /= max(va_b, 1)
    e0_ori_mae /= max(va_b, 1)
    print(f"[S2 Epoch 00/Pretrained] Val Loss: {e0_loss:.6f} | Disp MAE: {e0_disp_mae:.5f} | Ori MAE: {e0_ori_mae:.5f}")

    best_val_loss = e0_loss
    best_epoch = 0
    patience_counter = 0
    history = [{
        "epoch": 0,
        "train_loss": float("nan"),
        "val_loss": e0_loss,
        "val_disp_mae": e0_disp_mae,
        "val_ori_mae": e0_ori_mae,
        "lr": learning_rate,
    }]

    # Save initial pretrained state as fallback best
    best_state = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": e0_loss,
        "val_disp_mae": e0_disp_mae,
        "val_ori_mae": e0_ori_mae,
        "config": {
            "model": "TurningSpecialistS2",
            "in_channels": 6,
            "conv_channels": 32,
            "gru_hidden": 32,
            "num_gru_layers": 1,
            "dropout": 0.20,
            "use_multihead_attention": True,
            "use_cross_task_coupling": True,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    torch.save(best_state, CKPT_DIR / "best_specialist_S2.pth")

    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        total_loss, total_ld, total_lo, total_lz = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for X, yd, yo, yz in tr_loader:
            optimizer.zero_grad()
            d_pred, o_pred, z_logit = model(X)

            l_d = loss_fn_huber(d_pred, yd)
            l_o = loss_fn_huber(o_pred, yo)
            l_z = loss_fn_zupt(z_logit, yz)
            loss = l_d + alpha_ori * l_o + beta_zupt * l_z

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ld += l_d.item()
            total_lo += l_o.item()
            total_lz += l_z.item()
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)
        tr_loss = total_loss / nb

        # ── Validation (Strictly on Held-Out High-Yaw Slice) ──
        model.eval()
        va_loss, va_disp_mae, va_ori_mae = 0.0, 0.0, 0.0
        va_batches = 0
        with torch.no_grad():
            for X, yd, yo, yz in va_loader:
                d_pred, o_pred, z_logit = model(X)
                l_d = loss_fn_huber(d_pred, yd)
                l_o = loss_fn_huber(o_pred, yo)
                l_z = loss_fn_zupt(z_logit, yz)
                loss = l_d + alpha_ori * l_o + beta_zupt * l_z

                va_loss += loss.item()
                va_disp_mae += nn.functional.l1_loss(d_pred, yd).item()
                va_ori_mae += nn.functional.l1_loss(o_pred, yo).item()
                va_batches += 1

        nva = max(va_batches, 1)
        va_loss /= nva
        va_disp_mae /= nva
        va_ori_mae /= nva

        lr_curr = optimizer.param_groups[0]["lr"]
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_disp_mae": va_disp_mae,
            "val_ori_mae": va_ori_mae,
            "lr": lr_curr,
        }
        history.append(rec)

        improved = va_loss < best_val_loss
        star = " *" if improved else ""
        print(
            f"[S2 Epoch {epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {tr_loss:.6f} | "
            f"Val Loss (High-Yaw): {va_loss:.6f} | "
            f"Disp MAE: {va_disp_mae:.5f} | "
            f"Ori MAE: {va_ori_mae:.5f} | "
            f"LR: {lr_curr:.6f}{star}"
        )

        if improved:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va_loss,
                "val_disp_mae": va_disp_mae,
                "val_ori_mae": va_ori_mae,
                "config": {
                    "model": "TurningSpecialistS2",
                    "in_channels": 6,
                    "conv_channels": 32,
                    "gru_hidden": 32,
                    "num_gru_layers": 1,
                    "dropout": 0.20,
                    "use_multihead_attention": True,
                    "use_cross_task_coupling": True,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            }
            torch.save(best_state, CKPT_DIR / "best_specialist_S2.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[S2] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    elapsed = time.time() - t_start
    print(f"[S2] Finished in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")

    # Load best weights for export
    ckpt = torch.load(CKPT_DIR / "best_specialist_S2.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export_onnx_specialist(model, str(CKPT_DIR / "best_specialist_S2.onnx"), in_channels=6, device=device)
    export_torchscript_specialist(model, str(CKPT_DIR / "best_specialist_S2_torchscript.pt"), in_channels=6, device=device)

    with open(RESULTS_DIR / "training_history_S2.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_val_loss


def train_both(max_epochs: int = 50, patience: int = 12):
    t0 = time.time()
    train_specialist_s1(max_epochs=max_epochs, patience=patience)
    train_specialist_s2(max_epochs=max_epochs, patience=patience)
    print(f"\n[v7] Dual-Specialist fine-tuning complete in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINO-DR v7 Specialists")
    parser.add_argument("--specialist", choices=["S1", "S2", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    if args.specialist == "S1":
        train_specialist_s1(max_epochs=args.epochs, patience=args.patience)
    elif args.specialist == "S2":
        train_specialist_s2(max_epochs=args.epochs, patience=args.patience)
    else:
        train_both(max_epochs=args.epochs, patience=args.patience)
