"""
train_v4.py
-----------
Production training engine for PINO-DR v4 (Turn-Focused).

Key features:
- Multi-task loss: MAE (displacement) + Turn-Adaptive Huber (orientation) + BCE (ZUPT) + Centripetal Physics loss.
- Turn-adaptive orientation weighting based on composite turn score sqrt((w/sigma_w)^2 + (w_dot/sigma_w_dot)^2).
- Soft physical centripetal regularizer L_physics = Huber(v_pred * w_pred, a_lat_measured) in physical units (m/s^2).
- CosineAnnealingWarmRestarts scheduler (T_0=25, T_mult=2).
- Early stopping with configurable patience (default 15 epochs) up to max 80 epochs.
- Per-epoch checkpoint saving in checkpoints/epochs/epoch_XXX.pth.
- Best model tracking saved to checkpoints/best_model.pth.
- Supports full ablation sequence (A through E).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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

from src.models_v4 import (
    PINODeadReckoningNetV4,
    export_onnx_v4,
    export_torchscript_v4,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
EPOCH_DIR = CKPT_DIR / "epochs"
RESULTS_DIR = ROOT / "results"

DEFAULT_CONFIG = {
    "max_epochs": 80,
    "patience": 15,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "alpha_base": 1.0,
    "alpha_turn": 2.0,
    "beta_zupt": 0.25,
    "lambda_physics": 0.05,
    "scheduler_T0": 25,
    "scheduler_Tmult": 2,
    # Architecture
    "in_channels": 6,
    "conv_channels": 32,
    "gru_hidden": 32,
    "num_gru_layers": 1,
    "dropout": 0.20,
    "use_multihead_attention": True,
    "use_cross_task_coupling": True,
    "use_turn_adaptive_loss": True,
    "use_physics_loss": True,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load preprocessed v4 data and scalers."""
    npz_path = DATA_DIR / "dataset_splits_v4.npz"
    scaler_path = DATA_DIR / "scalers_v4.pkl"
    if not npz_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"v4 cache missing. Run `python -m src.preprocess_v4` first.\n"
            f"Need {npz_path} and {scaler_path}"
        )
    d = np.load(npz_path)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    return d, scalers


def build_loaders(d, in_channels: int, batch_size: int, device: torch.device):
    """Build DataLoaders for train, val, test."""
    def _make(X_key, yd_key, yo_key, yz_key, shuffle):
        X_full = d[X_key][:, :, :in_channels]
        X = torch.tensor(X_full, dtype=torch.float32).to(device)
        yd = torch.tensor(d[yd_key], dtype=torch.float32).to(device)
        yo = torch.tensor(d[yo_key], dtype=torch.float32).to(device)
        yz = torch.tensor(d[yz_key], dtype=torch.float32).to(device)
        ds = TensorDataset(X, yd, yo, yz)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    tr = _make("X_tr", "y_d_tr", "y_o_tr", "y_z_tr", shuffle=True)
    va = _make("X_va", "y_d_va", "y_o_va", "y_z_va", shuffle=False)
    te = _make("X_te", "y_d_te", "y_o_te", "y_z_te", shuffle=False)
    return tr, va, te


def compute_turn_weights(X: torch.Tensor, scalers: dict, alpha_base: float, alpha_turn: float) -> torch.Tensor:
    """Compute per-sample orientation loss weight based on turn score."""
    turn_stats = scalers["turn_stats"]
    std_w = turn_stats["std_w_yaw"]
    std_w_dot = turn_stats["std_w_accel"]
    tau_turn = turn_stats["tau_turn"]

    # In scalers: w_yaw is channel 1, w_yaw_accel is channel 4
    # Unscale using known physical limits: w_yaw in [-1.0, 1.0], w_accel in [-2.0, 2.0]
    w_yaw_scaled = X[:, -1, 1:2]
    w_yaw_phys = w_yaw_scaled * 2.0 - 1.0  # approximate physical rad/s

    if X.shape[-1] >= 5:
        w_accel_scaled = X[:, -1, 4:5]
        w_accel_phys = w_accel_scaled * 4.0 - 2.0  # approximate physical rad/s^2
    else:
        w_accel_phys = torch.zeros_like(w_yaw_phys)

    turn_score = torch.sqrt((w_yaw_phys / std_w)**2 + (w_accel_phys / std_w_dot)**2)
    weights = torch.where(turn_score > tau_turn, alpha_turn, alpha_base)
    return weights


def compute_physics_loss(d_pred: torch.Tensor, o_pred: torch.Tensor, X: torch.Tensor, scalers: dict) -> torch.Tensor:
    """
    Computes Huber(a_lat_measured, v_pred * w_pred) in physical SI units (m/s^2).
    """
    v_max = scalers["y_disp"].data_max_[0]
    v_min = scalers["y_disp"].data_min_[0]
    w_max = scalers["y_ori"].data_max_[0]
    w_min = scalers["y_ori"].data_min_[0]

    # Model predictions in physical units
    v_phys = d_pred * (v_max - v_min) + v_min         # m/s
    w_phys = o_pred * (w_max - w_min) + w_min         # rad/s
    centripetal_pred = v_phys * w_phys                # m/s^2

    # Measured a_lat from channel 2 (unscaled, approx [-8.0, 8.0])
    idx_alat = 9 * X.shape[-1] + 2
    a_min = scalers["X"].data_min_[idx_alat]
    a_max = scalers["X"].data_max_[idx_alat]
    a_lat_measured = X[:, -1, 2:3] * (a_max - a_min) + a_min

    # Huber loss with delta=1.0 m/s^2
    huber_phys = nn.functional.huber_loss(centripetal_pred, a_lat_measured, delta=1.0, reduction='mean')
    return huber_phys


def train_one_epoch(model, loader, optimizer, loss_fn_disp, loss_fn_ori_elementwise,
                    loss_fn_zupt, cfg, scalers, device):
    model.train()
    total_loss = 0.0
    total_disp_loss = 0.0
    total_ori_loss = 0.0
    total_zupt_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0

    for X, yd, yo, yz in loader:
        optimizer.zero_grad()
        d_pred, o_pred, z_logit = model(X)

        l_disp = loss_fn_disp(d_pred, yd)

        # Orientation loss (turn-adaptive or fixed)
        if cfg["use_turn_adaptive_loss"] and X.shape[-1] >= 5:
            weights = compute_turn_weights(X, scalers, cfg["alpha_base"], cfg["alpha_turn"])
            l_ori_elem = loss_fn_ori_elementwise(o_pred, yo)
            l_ori = torch.mean(weights * l_ori_elem)
        else:
            l_ori = cfg["alpha_base"] * torch.mean(loss_fn_ori_elementwise(o_pred, yo))

        l_zupt = loss_fn_zupt(z_logit, yz)

        # Centripetal physics regularizer
        if cfg["use_physics_loss"] and cfg["lambda_physics"] > 0:
            l_phys = compute_physics_loss(d_pred, o_pred, X, scalers)
        else:
            l_phys = torch.tensor(0.0, device=device)

        loss = l_disp + l_ori + cfg["beta_zupt"] * l_zupt + cfg["lambda_physics"] * l_phys

        loss.backward()
        if cfg["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        total_loss += loss.item()
        total_disp_loss += l_disp.item()
        total_ori_loss += l_ori.item()
        total_zupt_loss += l_zupt.item()
        total_phys_loss += l_phys.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_disp_loss / n, total_ori_loss / n, total_phys_loss / n


@torch.no_grad()
def validate(model, loader, loss_fn_disp, loss_fn_ori_elementwise, loss_fn_zupt, cfg, scalers, device):
    model.eval()
    total_loss = 0.0
    total_disp_mae = 0.0
    total_ori_mae = 0.0
    total_turn_ori_mae = 0.0
    turn_samples = 0
    n_batches = 0

    for X, yd, yo, yz in loader:
        d_pred, o_pred, z_logit = model(X)

        l_disp = loss_fn_disp(d_pred, yd)

        if cfg["use_turn_adaptive_loss"] and X.shape[-1] >= 5:
            weights = compute_turn_weights(X, scalers, cfg["alpha_base"], cfg["alpha_turn"])
            l_ori_elem = loss_fn_ori_elementwise(o_pred, yo)
            l_ori = torch.mean(weights * l_ori_elem)
        else:
            l_ori = cfg["alpha_base"] * torch.mean(loss_fn_ori_elementwise(o_pred, yo))

        l_zupt = loss_fn_zupt(z_logit, yz)

        if cfg["use_physics_loss"] and cfg["lambda_physics"] > 0:
            l_phys = compute_physics_loss(d_pred, o_pred, X, scalers)
        else:
            l_phys = torch.tensor(0.0, device=device)

        loss = l_disp + l_ori + cfg["beta_zupt"] * l_zupt + cfg["lambda_physics"] * l_phys

        total_loss += loss.item()
        disp_err = nn.functional.l1_loss(d_pred, yd, reduction='none')
        ori_err = nn.functional.l1_loss(o_pred, yo, reduction='none')

        total_disp_mae += disp_err.mean().item()
        total_ori_mae += ori_err.mean().item()

        # Track turn-specific orientation error
        if X.shape[-1] >= 5:
            weights = compute_turn_weights(X, scalers, 0.0, 1.0)
            is_turn = weights > 0.5
            if is_turn.any():
                total_turn_ori_mae += (ori_err * is_turn).sum().item()
                turn_samples += is_turn.sum().item()

        n_batches += 1

    n = max(n_batches, 1)
    turn_ori_mae = (total_turn_ori_mae / max(turn_samples, 1)) if turn_samples > 0 else (total_ori_mae / n)
    return total_loss / n, total_disp_mae / n, total_ori_mae / n, turn_ori_mae


def run_training(config_overrides: dict = None, tag: str = "v4_full"):
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[v4][{tag}] Training on: {device}")
    print(f"[v4][{tag}] Configuration: in_ch={cfg['in_channels']}, "
          f"multihead_attn={cfg['use_multihead_attention']}, coupling={cfg['use_cross_task_coupling']}, "
          f"turn_loss={cfg['use_turn_adaptive_loss']}, lambda_phys={cfg['lambda_physics']}")

    d, scalers = load_data()
    tr_loader, va_loader, te_loader = build_loaders(d, cfg["in_channels"], cfg["batch_size"], device)

    model = PINODeadReckoningNetV4(
        in_channels=cfg["in_channels"],
        conv_channels=cfg["conv_channels"],
        gru_hidden=cfg["gru_hidden"],
        num_gru_layers=cfg["num_gru_layers"],
        dropout=cfg["dropout"],
        use_multihead_attention=cfg["use_multihead_attention"],
        use_cross_task_coupling=cfg["use_cross_task_coupling"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v4][{tag}] Model parameters: {n_params:,} (budget <= 25,000: {'PASS' if n_params <= 25000 else 'FAIL'})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["scheduler_T0"], T_mult=cfg["scheduler_Tmult"]
    )

    loss_fn_disp = nn.L1Loss()
    loss_fn_ori_elem = nn.HuberLoss(delta=0.1, reduction='none')
    loss_fn_zupt = nn.BCEWithLogitsLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]

    history = {
        "train_loss": [], "val_loss": [], "val_disp_mae": [],
        "val_ori_mae": [], "val_turn_ori_mae": [], "train_phys_loss": [], "lr": []
    }

    t_start = time.time()
    print(f"[v4][{tag}] Max epochs: {max_epochs}, Early stopping patience: {patience}\n")

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_d_loss, tr_o_loss, tr_phys_loss = train_one_epoch(
            model, tr_loader, optimizer,
            loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
            cfg, scalers, device
        )

        val_loss, val_d_mae, val_o_mae, val_turn_o_mae = validate(
            model, va_loader,
            loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
            cfg, scalers, device
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_disp_mae"].append(val_d_mae)
        history["val_ori_mae"].append(val_o_mae)
        history["val_turn_ori_mae"].append(val_turn_o_mae)
        history["train_phys_loss"].append(tr_phys_loss)
        history["lr"].append(lr)

        # Per-epoch checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_disp_mae": val_d_mae,
            "val_ori_mae": val_o_mae,
            "val_turn_ori_mae": val_turn_o_mae,
            "config": cfg,
        }
        torch.save(ckpt, EPOCH_DIR / f"epoch_{epoch:03d}.pth")

        # Best model tracking
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_ckpt_path = CKPT_DIR / (f"best_model_{tag}.pth" if tag != "v4_full" else "best_model.pth")
            torch.save(ckpt, best_ckpt_path)
        else:
            epochs_no_improve += 1

        dt = time.time() - t_ep
        tag_str = " [BEST]" if is_best else ""
        print(
            f"  Epoch {epoch:2d}/{max_epochs} | "
            f"Train: {tr_loss:.5f} (Phys: {tr_phys_loss:.3f}) | "
            f"Val: {val_loss:.5f} | D-MAE: {val_d_mae:.4f} | "
            f"O-MAE: {val_o_mae:.4f} | Turn-O: {val_turn_o_mae:.4f} | "
            f"LR: {lr:.2e} | {dt:.1f}s{tag_str}"
        )

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n[v4][{tag}] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_time = time.time() - t_start
    print(f"\n[v4][{tag}] Training finished in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[v4][{tag}] Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")

    # Reload best checkpoint
    best_file = CKPT_DIR / (f"best_model_{tag}.pth" if tag != "v4_full" else "best_model.pth")
    best_ckpt = torch.load(best_file, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Test set evaluation
    _, test_d_mae, test_o_mae, test_turn_o_mae = validate(
        model, te_loader,
        loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
        cfg, scalers, device
    )
    print(f"[v4][{tag}] Test MAE — Disp: {test_d_mae:.5f}, Ori: {test_o_mae:.5f}, Turn-Ori: {test_turn_o_mae:.5f}")

    if tag == "v4_full":
        # Export ONNX & TorchScript
        export_onnx_v4(model, str(CKPT_DIR / "best_model.onnx"), device, in_channels=cfg["in_channels"])
        export_torchscript_v4(model, str(CKPT_DIR / "best_model_torchscript.pt"), device, in_channels=cfg["in_channels"])

        # Export final_model_v4.pkl
        bundle = {
            "model_state_dict": model.state_dict(),
            "model_config": cfg,
            "scalers": scalers,
            "metadata": {
                "architecture": "PINO-DR v4 (Turn-Focused: 6ch + 2-Head Directional Attn + Predictive Coupling)",
                "parameters": n_params,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_disp_mae": test_d_mae,
                "test_ori_mae": test_o_mae,
                "test_turn_ori_mae": test_turn_o_mae,
                "total_training_time_s": total_time,
                "dataset": "IO-VNBD (smartphone IMU)",
                "split": "frozen 70/10/20 trip-level before windowing",
            }
        }
        with open(CKPT_DIR / "final_model_v4.pkl", "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[v4] Saved {CKPT_DIR / 'final_model_v4.pkl'}")

    # Save history
    hist_path = RESULTS_DIR / f"training_history_{tag}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_disp_mae": test_d_mae,
        "test_ori_mae": test_o_mae,
        "test_turn_ori_mae": test_turn_o_mae,
        "n_params": n_params,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PINO-DR v4")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D", "E", "all"], default="E",
                        help="A: V3 Baseline, B: +YawAccel, C: +CentripetalRes, D: +Attn/Coupling, E: Full V4")
    args = parser.parse_args()

    if args.ablation == "E":
        run_training(tag="v4_full")
    elif args.ablation == "A":
        # V3 Baseline configuration (4 channels, single attention, no coupling, standard loss)
        run_training({
            "in_channels": 4,
            "use_multihead_attention": False,
            "use_cross_task_coupling": False,
            "use_turn_adaptive_loss": False,
            "use_physics_loss": False,
            "alpha_base": 1.2,
        }, tag="ablation_A_v3_baseline")
    elif args.ablation == "B":
        # +Yaw Acceleration (5 channels)
        run_training({
            "in_channels": 5,
            "use_multihead_attention": False,
            "use_cross_task_coupling": False,
            "use_turn_adaptive_loss": False,
            "use_physics_loss": False,
            "alpha_base": 1.2,
        }, tag="ablation_B_yaw_accel")
    elif args.ablation == "C":
        # +Centripetal Residual (6 channels)
        run_training({
            "in_channels": 6,
            "use_multihead_attention": False,
            "use_cross_task_coupling": False,
            "use_turn_adaptive_loss": False,
            "use_physics_loss": False,
            "alpha_base": 1.2,
        }, tag="ablation_C_centripetal_res")
    elif args.ablation == "D":
        # +2-Head Attention & Cross-Task Coupling (6 channels)
        run_training({
            "in_channels": 6,
            "use_multihead_attention": True,
            "use_cross_task_coupling": True,
            "use_turn_adaptive_loss": False,
            "use_physics_loss": False,
            "alpha_base": 1.2,
        }, tag="ablation_D_attn_coupling")


if __name__ == "__main__":
    main()
