"""
train_v5.py
-----------
Training engine for PINO-DR v5.

Step 1: Experiment 1(a) — Unmodified Model D architecture with 7 input channels
(adding raw rear wheel speed v_wheel as channel 6).
"""

from __future__ import annotations

import argparse
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

from v5_wheel_aided.src.models_v5 import (
    PINODeadReckoningNetV5,
    export_onnx_v5,
    export_torchscript_v5,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
EPOCH_DIR = CKPT_DIR / "epochs"
RESULTS_DIR = ROOT / "results"

DEFAULT_CONFIG = {
    "in_channels": 7,
    "conv_channels": 32,
    "gru_hidden": 32,
    "num_gru_layers": 1,
    "dropout": 0.20,
    "use_multihead_attention": True,
    "use_cross_task_coupling": True,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 80,
    "patience": 15,
    "grad_clip": 1.0,
    "scheduler_T0": 25,
    "scheduler_Tmult": 2,
    "alpha_base": 1.2,
    "beta_zupt": 0.5,
    "use_turn_adaptive_loss": False,
    "use_physics_loss": False,
    "lambda_physics": 0.0,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    npz_path = DATA_DIR / "dataset_splits_v5.npz"
    scalers_path = DATA_DIR / "scalers_v5.pkl"
    if not npz_path.exists() or not scalers_path.exists():
        raise FileNotFoundError(f"Missing preprocessed data. Run preprocess_v5.py first.")
    d = np.load(npz_path)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    return d, scalers


def build_loaders(d, in_channels: int, batch_size: int, device: torch.device):
    def make(X, yd, yo, yz, shuffle):
        X_sub = X[:, :, :in_channels]
        ds = TensorDataset(
            torch.tensor(X_sub, dtype=torch.float32),
            torch.tensor(yd, dtype=torch.float32),
            torch.tensor(yo, dtype=torch.float32),
            torch.tensor(yz, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

    tr = make(d["X_train"], d["y_disp_train"], d["y_ori_train"], d["y_zupt_train"], shuffle=True)
    va = make(d["X_val"], d["y_disp_val"], d["y_ori_val"], d["y_zupt_val"], shuffle=False)
    te = make(d["X_test"], d["y_disp_test"], d["y_ori_test"], d["y_zupt_test"], shuffle=False)
    return tr, va, te


def train_one_epoch(model, loader, optimizer, loss_fn_disp, loss_fn_ori_elementwise, loss_fn_zupt, cfg, scalers, device):
    model.train()
    total_loss = 0.0
    total_disp_loss = 0.0
    total_ori_loss = 0.0
    total_zupt_loss = 0.0
    n_batches = 0

    for X, yd, yo, yz in loader:
        X = X.to(device)
        yd = yd.to(device)
        yo = yo.to(device)
        yz = yz.to(device)

        optimizer.zero_grad()
        d_pred, o_pred, z_logit = model(X)

        l_disp = loss_fn_disp(d_pred, yd)
        l_ori = cfg["alpha_base"] * torch.mean(loss_fn_ori_elementwise(o_pred, yo))
        l_zupt = loss_fn_zupt(z_logit, yz)

        loss = l_disp + l_ori + cfg["beta_zupt"] * l_zupt

        loss.backward()
        if cfg["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        total_loss += loss.item()
        total_disp_loss += l_disp.item()
        total_ori_loss += l_ori.item()
        total_zupt_loss += l_zupt.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_disp_loss / n, total_ori_loss / n


@torch.no_grad()
def validate(model, loader, loss_fn_disp, loss_fn_ori_elementwise, loss_fn_zupt, cfg, scalers, device):
    model.eval()
    total_loss = 0.0
    total_disp_mae = 0.0
    total_ori_mae = 0.0
    n_batches = 0

    for X, yd, yo, yz in loader:
        X = X.to(device)
        yd = yd.to(device)
        yo = yo.to(device)
        yz = yz.to(device)

        d_pred, o_pred, z_logit = model(X)

        l_disp = loss_fn_disp(d_pred, yd)
        l_ori = cfg["alpha_base"] * torch.mean(loss_fn_ori_elementwise(o_pred, yo))
        l_zupt = loss_fn_zupt(z_logit, yz)

        loss = l_disp + l_ori + cfg["beta_zupt"] * l_zupt

        total_loss += loss.item()
        disp_err = nn.functional.l1_loss(d_pred, yd, reduction='none')
        ori_err = nn.functional.l1_loss(o_pred, yo, reduction='none')

        total_disp_mae += disp_err.mean().item()
        total_ori_mae += ori_err.mean().item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_disp_mae / n, total_ori_mae / n


def run_training(config_overrides: dict = None, tag: str = "v5_step1_raw_wheel"):
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[v5][{tag}] Training on: {device}", flush=True)
    print(f"[v5][{tag}] Configuration: in_ch={cfg['in_channels']}, "
          f"multihead_attn={cfg['use_multihead_attention']}, coupling={cfg['use_cross_task_coupling']}", flush=True)

    d, scalers = load_data()
    tr_loader, va_loader, te_loader = build_loaders(d, cfg["in_channels"], cfg["batch_size"], device)

    model = PINODeadReckoningNetV5(
        in_channels=cfg["in_channels"],
        conv_channels=cfg["conv_channels"],
        gru_hidden=cfg["gru_hidden"],
        num_gru_layers=cfg["num_gru_layers"],
        dropout=cfg["dropout"],
        use_multihead_attention=cfg["use_multihead_attention"],
        use_cross_task_coupling=cfg["use_cross_task_coupling"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v5][{tag}] Model parameters: {n_params:,} (budget <= 25,000: {'PASS' if n_params <= 25000 else 'FAIL'})", flush=True)

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
        "train_loss": [],
        "val_loss": [],
        "val_disp_mae": [],
        "val_ori_mae": [],
        "lr": [],
    }

    t_start = time.time()

    for epoch in range(max_epochs):
        t_ep0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_d_loss, tr_o_loss = train_one_epoch(
            model, tr_loader, optimizer,
            loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
            cfg, scalers, device
        )

        val_loss, val_d_mae, val_o_mae = validate(
            model, va_loader,
            loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
            cfg, scalers, device
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_disp_mae"].append(val_d_mae)
        history["val_ori_mae"].append(val_o_mae)
        history["lr"].append(lr)

        # Best model tracking
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_file = CKPT_DIR / f"best_model_{tag}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_disp_mae": val_d_mae,
                "val_ori_mae": val_o_mae,
                "config": cfg,
                "scalers": scalers,
            }, best_file)
        else:
            epochs_no_improve += 1

        ep_time = time.time() - t_ep0
        star = " *" if is_best else ""
        if epoch % 5 == 0 or is_best or epoch == max_epochs - 1:
            print(f"[v5][{tag}] Ep {epoch:02d}/{max_epochs} | "
                  f"Train: {tr_loss:.5f} | Val: {val_loss:.5f} "
                  f"(D-MAE: {val_d_mae:.4f}, O-MAE: {val_o_mae:.4f}) | "
                  f"LR: {lr:.2e} | {ep_time:.1f}s{star}", flush=True)

        if epochs_no_improve >= patience:
            print(f"[v5][{tag}] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).", flush=True)
            break

    total_time = time.time() - t_start
    print(f"\n[v5][{tag}] Training completed in {total_time/60:.1f} min. Best epoch: {best_epoch} (val_loss: {best_val_loss:.6f})", flush=True)

    # Save training history
    with open(RESULTS_DIR / f"training_history_{tag}.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best weights for final test evaluation
    best_ckpt = torch.load(best_file, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Test set evaluation
    _, test_d_mae, test_o_mae = validate(
        model, te_loader,
        loss_fn_disp, loss_fn_ori_elem, loss_fn_zupt,
        cfg, scalers, device
    )
    print(f"[v5][{tag}] Test MAE — Disp: {test_d_mae:.5f}, Ori: {test_o_mae:.5f}", flush=True)

    return {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_disp_mae": test_d_mae,
        "test_ori_mae": test_o_mae,
        "n_params": n_params,
        "total_time_s": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINO-DR v5")
    parser.add_argument("--tag", default="v5_step1_raw_wheel", help="Experiment tag")
    args = parser.parse_args()
    run_training(tag=args.tag)
