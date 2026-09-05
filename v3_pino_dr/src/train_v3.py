"""
train_v3.py
-----------
Production training engine for PINO-DR v3.

Key design constraints (user-specified):
- NO early stopping — trains for the FULL epoch count.
- Per-epoch checkpoint saving to checkpoints_v3/epochs/epoch_XXX.pth.
- Best model tracking by validation loss → saved as best_model.pth.
- End-of-training: export best model as ONNX, TorchScript, and final_model_v3.pkl.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models_v3 import PINODeadReckoningNet, export_onnx_v3, export_torchscript_v3

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "preprocessed" / "v3"
CKPT_DIR = ROOT / "checkpoints_v3"
EPOCH_DIR = CKPT_DIR / "epochs"
RESULTS_DIR = ROOT / "results"

CONFIG = {
    "epochs": 60,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "alpha_ori": 1.2,     # high priority for orientation to preserve heading in 2D space
    "beta_zupt": 0.25,    # weight for ZUPT BCE loss
    "scheduler_T0": 20,
    "scheduler_Tmult": 2,
    # Model architecture
    "in_channels": 4,
    "conv_channels": 32,
    "gru_hidden": 32,
    "num_gru_layers": 1,
    "dropout": 0.20,
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
    """Load preprocessed v3 data and scalers."""
    npz_path = CACHE_DIR / "dataset_splits_v3.npz"
    scaler_path = CACHE_DIR / "scalers_v3.pkl"
    if not npz_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"v3 cache missing. Run `python -m src.preprocess_v3` first.\n"
            f"Need {npz_path} and {scaler_path}"
        )
    d = np.load(npz_path)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    return d, scalers


def build_loaders(d, batch_size: int, device: torch.device):
    """Build DataLoaders for train, val, test."""
    def _make(X_key, yd_key, yo_key, yz_key, shuffle):
        X = torch.tensor(d[X_key], dtype=torch.float32).to(device)
        yd = torch.tensor(d[yd_key], dtype=torch.float32).to(device)
        yo = torch.tensor(d[yo_key], dtype=torch.float32).to(device)
        yz = torch.tensor(d[yz_key], dtype=torch.float32).to(device)
        ds = TensorDataset(X, yd, yo, yz)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    tr = _make("X_tr", "y_d_tr", "y_o_tr", "y_z_tr", shuffle=True)
    va = _make("X_va", "y_d_va", "y_o_va", "y_z_va", shuffle=False)
    te = _make("X_te", "y_d_te", "y_o_te", "y_z_te", shuffle=False)
    return tr, va, te


def train_one_epoch(model, loader, optimizer, loss_fn_disp, loss_fn_ori,
                    loss_fn_zupt, alpha, beta, grad_clip, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X, yd, yo, yz in loader:
        optimizer.zero_grad()
        d_pred, o_pred, z_logit = model(X)

        l_disp = loss_fn_disp(d_pred, yd)
        l_ori = loss_fn_ori(o_pred, yo)
        l_zupt = loss_fn_zupt(z_logit, yz)

        loss = l_disp + alpha * l_ori + beta * l_zupt

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn_disp, loss_fn_ori, loss_fn_zupt, alpha, beta):
    model.eval()
    total_loss = 0.0
    total_disp_mae = 0.0
    total_ori_mae = 0.0
    n_batches = 0
    for X, yd, yo, yz in loader:
        d_pred, o_pred, z_logit = model(X)
        l_disp = loss_fn_disp(d_pred, yd)
        l_ori = loss_fn_ori(o_pred, yo)
        l_zupt = loss_fn_zupt(z_logit, yz)

        loss = l_disp + alpha * l_ori + beta * l_zupt
        total_loss += loss.item()
        total_disp_mae += nn.functional.l1_loss(d_pred, yd).item()
        total_ori_mae += nn.functional.l1_loss(o_pred, yo).item()
        n_batches += 1
    n = max(n_batches, 1)
    return total_loss / n, total_disp_mae / n, total_ori_mae / n


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v3] Training on: {device}")

    d, scalers = load_data()
    tr_loader, va_loader, te_loader = build_loaders(d, CONFIG["batch_size"], device)

    model = PINODeadReckoningNet(
        in_channels=CONFIG["in_channels"],
        conv_channels=CONFIG["conv_channels"],
        gru_hidden=CONFIG["gru_hidden"],
        num_gru_layers=CONFIG["num_gru_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v3] PINO-DR parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG["scheduler_T0"], T_mult=CONFIG["scheduler_Tmult"]
    )

    loss_fn_disp = nn.L1Loss()           # MAE for displacement
    loss_fn_ori = nn.HuberLoss(delta=0.1) # Huber for orientation
    loss_fn_zupt = nn.BCEWithLogitsLoss() # BCE for ZUPT classifier

    # ── Checkpoint directories ──
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_disp_mae": [],
               "val_ori_mae": [], "lr": []}

    t_start = time.time()
    epochs = CONFIG["epochs"]

    print(f"\n[v3] Training for {epochs} epochs (NO early stopping)...\n")

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, tr_loader, optimizer,
            loss_fn_disp, loss_fn_ori, loss_fn_zupt,
            CONFIG["alpha_ori"], CONFIG["beta_zupt"],
            CONFIG["grad_clip"], device,
        )
        val_loss, val_d_mae, val_o_mae = validate(
            model, va_loader,
            loss_fn_disp, loss_fn_ori, loss_fn_zupt,
            CONFIG["alpha_ori"], CONFIG["beta_zupt"],
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_disp_mae"].append(val_d_mae)
        history["val_ori_mae"].append(val_o_mae)
        history["lr"].append(lr)

        # ── Per-epoch checkpoint ──
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_disp_mae": val_d_mae,
            "val_ori_mae": val_o_mae,
            "config": CONFIG,
        }
        torch.save(ckpt, EPOCH_DIR / f"epoch_{epoch:03d}.pth")

        # ── Best model tracking ──
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(ckpt, CKPT_DIR / "best_model.pth")

        dt = time.time() - t_ep
        tag = " [BEST]" if is_best else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"D-MAE: {val_d_mae:.5f} | O-MAE: {val_o_mae:.5f} | "
            f"LR: {lr:.2e} | {dt:.1f}s{tag}"
        )

    total_time = time.time() - t_start
    print(f"\n[v3] Training complete: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[v3] Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")

    # ── Reload best model ──
    best_ckpt = torch.load(CKPT_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # ── Test set evaluation ──
    _, test_d_mae, test_o_mae = validate(
        model, te_loader,
        loss_fn_disp, loss_fn_ori, loss_fn_zupt,
        CONFIG["alpha_ori"], CONFIG["beta_zupt"],
    )
    print(f"[v3] Test MAE — Displacement: {test_d_mae:.6f}, Orientation: {test_o_mae:.6f}")

    # ── Export ONNX & TorchScript ──
    export_onnx_v3(model, str(CKPT_DIR / "best_model.onnx"), device)
    export_torchscript_v3(model, str(CKPT_DIR / "best_model_torchscript.pt"), device)

    # ── Export final_model_v3.pkl ──
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": CONFIG["in_channels"],
            "conv_channels": CONFIG["conv_channels"],
            "gru_hidden": CONFIG["gru_hidden"],
            "num_gru_layers": CONFIG["num_gru_layers"],
            "dropout": CONFIG["dropout"],
        },
        "scalers": scalers,
        "metadata": {
            "architecture": "PINO-DR (Conv1D + BiGRU + Temporal Attention)",
            "parameters": n_params,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_disp_mae": test_d_mae,
            "test_ori_mae": test_o_mae,
            "total_training_time_s": total_time,
            "epochs_trained": epochs,
            "dataset": "IO-VNBD (smartphone IMU, S- files)",
            "split": "70/10/20 trip-level before windowing",
            "warning": "Use ONNX for production deployment. PKL requires trusted source.",
        },
    }
    with open(CKPT_DIR / "final_model_v3.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_sz = (CKPT_DIR / "final_model_v3.pkl").stat().st_size / 1024
    print(f"[v3] Saved final_model_v3.pkl ({pkl_sz:.1f} KB)")

    # ── Save training history ──
    with open(RESULTS_DIR / "training_history_v3.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Training summary ──
    summary = {
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_disp_mae": test_d_mae,
        "test_ori_mae": test_o_mae,
        "total_time_s": total_time,
        "epochs": epochs,
    }
    with open(RESULTS_DIR / "training_summary_v3.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot training curves ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        ep_range = range(1, epochs + 1)

        axes[0, 0].plot(ep_range, history["train_loss"], label="Train Loss", color="tab:blue")
        axes[0, 0].plot(ep_range, history["val_loss"], label="Val Loss", color="tab:orange")
        axes[0, 0].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (ep {best_epoch})")
        axes[0, 0].set_title("Total Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ep_range, history["val_disp_mae"], color="tab:blue")
        axes[0, 1].set_title("Val Displacement MAE"); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(ep_range, history["val_ori_mae"], color="tab:green")
        axes[1, 0].set_title("Val Orientation MAE"); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(ep_range, history["lr"], color="tab:red")
        axes[1, 1].set_title("Learning Rate"); axes[1, 1].set_yscale("log"); axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"PINO-DR v3 Training ({epochs} epochs, best={best_epoch})", fontsize=14)
        fig.savefig(RESULTS_DIR / "training_curves_v3.png", dpi=200)
        plt.close(fig)
        print(f"[v3] Training curves saved to {RESULTS_DIR / 'training_curves_v3.png'}")
    except Exception as e:
        print(f"[v3] Could not plot training curves: {e}")

    print("[v3] Training pipeline complete.")


if __name__ == "__main__":
    main()
