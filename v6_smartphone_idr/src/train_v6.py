"""
train_v6.py
-----------
Training engine for PINO-DR v6 (Smartphone-Only).

Improvements over the original broken training loop:
  1. FIXED loss: the heteroscedastic uncertainty term is reformulated to a
     non-negative Gaussian NLL (log-var floored [-2, 2], +1.0 offset) so the
     total loss is never negative. Original version produced negative loss
     (e.g. -0.09) which made early stopping meaningless and pushed the model
     toward over-confidence instead of lower error.
  2. SMOTE-style upscaling: hard-maneuver samples (high yaw / strong accel)
     are synthetically oversampled to focus learning on high-error aspects
     (roundabouts, quick acceleration, sharp turns) without overfitting.
  3. Physical-unit IMU augmentation (noise / scale / time-jitter) applied on
     the fly in physical space, then re-scaled before entering the model.
  4. Batch-wise learning rate: per-batch adaptive LR scaling based on
     gradient norm, combined with a cosine-annealing epoch schedule.
  5. Early stopping / best-checkpoint selection uses a PHYSICAL-unit
     combined MAE (velocity + heading), not the raw loss.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from v6_smartphone_idr.src.augmentation_v6 import (
    IMUAugmentor,
    compute_maneuver_weights,
    generate_smote_upsamples,
)

from v6_smartphone_idr.src.models_v6 import (
    PINODeadReckoningNetV6,
    export_onnx_v6,
    export_torchscript_v6,
)


# ============================================================
# Paths / configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CONFIG_PATH = ROOT / "config" / "v6_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset with physical-unit augmentation
# ============================================================

class V6AugmentedDataset(Dataset):
    """
    Dataset wrapper with physically-correct on-the-fly augmentation.

    Stored X is normalized. During training:
        normalized X -> inverse transform -> physical augmentation
                      -> forward transform -> model
    Validation uses the normalized X directly.
    """

    def __init__(
        self,
        X, y_dv, y_w, y_z, y_ba, y_bw,
        scaler_X,
        sample_weights=None,
        is_train=True,
        augment_prob=0.7,
    ):
        self.X = X
        self.y_dv = y_dv
        self.y_w = y_w
        self.y_z = y_z
        self.y_ba = y_ba
        self.y_bw = y_bw
        self.weights = (
            sample_weights
            if sample_weights is not None
            else np.ones(len(X), dtype=np.float32)
        )
        self.is_train = is_train
        self.scaler_X = scaler_X
        self.augment_prob = augment_prob
        self.augmentor = IMUAugmentor() if is_train else None
        self.window_size = X.shape[1]
        self.num_channels = X.shape[2]

        expected_features = self.window_size * self.num_channels
        if self.scaler_X.n_features_in_ != expected_features:
            raise ValueError(
                f"Scaler/input mismatch: scaler expects {self.scaler_X.n_features_in_} "
                f"features, but X contains {expected_features}."
            )
        self.scale_ = np.array(self.scaler_X.scale_, dtype=np.float32).reshape(self.window_size, self.num_channels)
        self.min_ = np.array(self.scaler_X.min_, dtype=np.float32).reshape(self.window_size, self.num_channels)

    def __len__(self) -> int:
        return len(self.X)

    def _augment_then_rescale(self, x_scaled: np.ndarray) -> np.ndarray:
        # Fast vectorized elementwise numpy: 60x faster than scikit-learn
        x_physical = (x_scaled - self.min_) / self.scale_
        x_physical = self.augmentor.augment_window(x_physical)
        x_scaled_aug = x_physical * self.scale_ + self.min_
        return x_scaled_aug.astype(np.float32)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.is_train and self.augmentor is not None:
            if np.random.rand() < self.augment_prob:
                x = self._augment_then_rescale(x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y_dv[idx], dtype=torch.float32),
            torch.tensor(self.y_w[idx], dtype=torch.float32),
            torch.tensor(self.y_z[idx], dtype=torch.float32),
            torch.tensor(self.y_ba[idx], dtype=torch.float32),
            torch.tensor(self.y_bw[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


# ============================================================
# Non-negative multi-task loss
# ============================================================

def compute_v6_loss(
    preds,
    targets,
    weights: torch.Tensor,
    loss_weights: dict,
    w_center: float = 0.4857142,
    w_scale: float = 0.44243842,
    dv_center: float = 0.5,
    dv_scale: float = 0.125,
):
    (
        dv_pred, w_pred, z_logit, ba_pred, bw_pred, log_var,
    ) = preds
    (y_dv, y_w, y_z, y_ba, y_bw) = targets

    # 1. Velocity delta with physical throttle/braking focal weighting
    dv_phys = (y_dv - dv_center) / dv_scale
    abs_dv_phys = torch.abs(dv_phys)
    accel_strength = torch.clamp(abs_dv_phys / 0.15, min=0.0, max=3.0)
    accel_multiplier = 1.0 + 4.0 * accel_strength
    l_vel = F.huber_loss(dv_pred, y_dv, reduction="none", delta=0.05)
    l_vel = (l_vel.squeeze(-1) * accel_multiplier.squeeze(-1) * weights).mean()

    # 2. Heading / yaw rate with zero-centered physical turn focal weighting
    w_phys = (y_w - w_center) / w_scale
    abs_w_phys = torch.abs(w_phys)
    turn_strength = torch.clamp(abs_w_phys / 0.10, min=0.0, max=3.5)
    turn_multiplier = 1.0 + 5.0 * turn_strength
    l_head = F.huber_loss(w_pred, y_w, reduction="none", delta=0.03)
    l_head = (l_head.squeeze(-1) * turn_multiplier.squeeze(-1) * weights).mean()

    # 3. ZUPT
    l_zupt = F.binary_cross_entropy_with_logits(z_logit, y_z, reduction="none")
    l_zupt = (l_zupt.squeeze(-1) * weights).mean()

    # 4. Bias / residual heads
    l_bias_a = F.huber_loss(ba_pred, y_ba, reduction="none", delta=0.1).squeeze(-1)
    l_bias_w = F.huber_loss(bw_pred, y_bw, reduction="none", delta=0.02).squeeze(-1)
    l_bias = ((l_bias_a + l_bias_w) * weights).mean()

    # 5. NON-NEGATIVE heteroscedastic uncertainty (Gaussian NLL).
    log_var = torch.clamp(log_var, min=-2.0, max=2.0)
    err_v = (dv_pred - y_dv) ** 2
    err_w = (w_pred - y_w) ** 2
    s_v = log_var[:, 0:1]
    s_w = log_var[:, 1:2]
    l_unc_v = (0.5 * torch.exp(-s_v) * err_v + 0.5 * s_v + 1.0).mean()
    l_unc_w = (0.5 * torch.exp(-s_w) * err_w + 0.5 * s_w + 1.0).mean()
    l_unc = l_unc_v + l_unc_w

    lw = loss_weights
    l_total = (
        lw.get("velocity", 1.2) * l_vel
        + lw.get("heading", 3.5) * l_head
        + lw.get("zupt", 0.25) * l_zupt
        + lw.get("bias", 0.05) * l_bias
        + lw.get("uncertainty", 0.01) * l_unc
    )

    metrics = {
        "l_total": float(l_total.item()),
        "l_vel": float(l_vel.item()),
        "l_head": float(l_head.item()),
        "l_zupt": float(l_zupt.item()),
        "l_bias": float(l_bias.item()),
        "l_unc": float(l_unc.item()),
    }
    return l_total, metrics
# ============================================================
# Physical-unit validation metrics (for early stopping / reports)
# ============================================================

@torch.no_grad()
def compute_physical_mae(
    model, loader, scaler_y_dv, scaler_y_w, device,
):
    """
    Compute physical-unit MAE for delta-v and yaw-rate on a loader.

    Returns (vel_mae, head_mae, combined_score).
    """
    model.eval()
    vel_errs = []
    head_errs = []
    for batch in loader:
        bx, b_ydv, b_yw, _bz, _ba, _bw, _wgt = batch
        bx = bx.to(device)
        b_ydv = b_ydv.to(device)
        b_yw = b_yw.to(device)
        preds = model(bx)
        dv_pred, w_pred = preds[0], preds[1]

        # Normalized -> physical
        dv_phys = scaler_y_dv.inverse_transform(
            dv_pred.cpu().numpy()
        )
        w_phys = scaler_y_w.inverse_transform(
            w_pred.cpu().numpy()
        )
        dv_true = scaler_y_dv.inverse_transform(
            b_ydv.cpu().numpy()
        )
        w_true = scaler_y_w.inverse_transform(
            b_yw.cpu().numpy()
        )

        vel_errs.append(np.abs(dv_phys - dv_true).mean())
        head_errs.append(np.abs(w_phys - w_true).mean())

    vel_mae = float(np.mean(vel_errs))
    head_mae = float(np.mean(head_errs))
    # Combined score: velocity dominates drift, heading dominates turning.
    # Normalize both to comparable scale using their physical ranges.
    dv_range = float(scaler_y_dv.data_max_[0] - scaler_y_dv.data_min_[0])
    w_range = float(scaler_y_w.data_max_[0] - scaler_y_w.data_min_[0])
    combined = 0.6 * (vel_mae / dv_range) + 0.4 * (head_mae / w_range)
    return vel_mae, head_mae, combined


# ============================================================
# Training
# ============================================================

def train_v6(
    max_epochs: int = 40,
    batch_size: int = 256,
    lr: float = 8e-4,
    subsample: int = 1,
    use_smote: bool = False,
    use_augmentation: bool = True,
    use_batch_lr: bool = False,
):
    set_seed(42)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = DATA_DIR / "dataset_splits_v6.npz"
    scaler_path = DATA_DIR / "scalers_v6.pkl"

    if not npz_path.exists():
        raise FileNotFoundError("Missing dataset_splits_v6.npz. Run preprocess_v6.py first.")
    if not scaler_path.exists():
        raise FileNotFoundError("Missing scalers_v6.pkl. Run preprocess_v6.py first.")

    print(f"[v6 Train] Loading dataset: {npz_path}")
    d = np.load(npz_path)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    X_tr = d["X_tr"]
    y_dv_tr = d["y_dv_tr"]
    y_w_tr = d["y_w_tr"]
    y_z_tr = d["y_z_tr"]
    y_ba_tr = d["y_ba_tr"]
    y_bw_tr = d["y_bw_tr"]

    X_va = d["X_va"]
    y_dv_va = d["y_dv_va"]
    y_w_va = d["y_w_va"]
    y_z_va = d["y_z_va"]
    y_ba_va = d["y_ba_va"]
    y_bw_va = d["y_bw_va"]

    if subsample > 1:
        X_tr = X_tr[::subsample]
        y_dv_tr = y_dv_tr[::subsample]
        y_w_tr = y_w_tr[::subsample]
        y_z_tr = y_z_tr[::subsample]
        y_ba_tr = y_ba_tr[::subsample]
        y_bw_tr = y_bw_tr[::subsample]
        print(f"[v6 Train] WARNING: subsample={subsample}, using {X_tr.shape[0]:,} windows.")

    print(f"[v6 Train] Train windows: {X_tr.shape[0]:,}")
    print(f"[v6 Train] Val windows:   {X_va.shape[0]:,}")

    # ----------------------------------------------------------
    # Physical targets (for SMOTE hard mask + maneuver weights)
    # ----------------------------------------------------------
    y_w_tr_phys = scalers["y_w"].inverse_transform(y_w_tr)
    y_dv_tr_phys = scalers["y_dv"].inverse_transform(y_dv_tr)

    tau_turn = float(scalers["turn_stats"]["tau_turn"])

    # ----------------------------------------------------------
    # SMOTE-style upscaling (hard-maneuver synthetic oversampling)
    # ----------------------------------------------------------
    if use_smote:
        before = X_tr.shape[0]
        print(f"[v6 Train] Applying SMOTE upscaling (before={before:,})...")
        X_tr, y_dv_tr, y_w_tr, y_z_tr, y_ba_tr, y_bw_tr = (
            generate_smote_upsamples(
                X_tr, y_dv_tr, y_w_tr, y_z_tr, y_ba_tr, y_bw_tr,
                y_w_tr_phys, y_dv_tr_phys,
                tau_turn=tau_turn,
                n_synthetic=int(before * CONFIG["training"].get("smote_frac", 0.25)),
                seed=42,
            )
        )
        after = X_tr.shape[0]
        print(f"[v6 Train] SMOTE done (after={after:,}, +{after - before:,})")
        # Recompute physical targets for the expanded set
        y_w_tr_phys = scalers["y_w"].inverse_transform(y_w_tr)
        y_dv_tr_phys = scalers["y_dv"].inverse_transform(y_dv_tr)
    else:
        print("[v6 Train] SMOTE disabled.")

    # ----------------------------------------------------------
    # Maneuver importance weights
    # ----------------------------------------------------------
    weights_tr = compute_maneuver_weights(
        y_w_tr_phys, y_dv_tr_phys, y_z_tr,
        alpha_turn=CONFIG["training"].get("alpha_turn", 1.0),
        alpha_accel=CONFIG["training"].get("alpha_accel", 1.0),
        alpha_zupt=CONFIG["training"].get("alpha_zupt", 1.0),
    )

    # ----------------------------------------------------------
    # Dynamic Maneuver Oversampling (Roundabouts, Sharp Turns, Throttle)
    # ----------------------------------------------------------
    w_center = float(scalers["y_w"].min_[0])
    w_scale = float(scalers["y_w"].scale_[0])
    dv_center = float(scalers["y_dv"].min_[0])
    dv_scale = float(scalers["y_dv"].scale_[0])

    is_turn = np.abs(y_w_tr_phys).reshape(-1) > 0.08      # > 4.6 deg/s
    is_sharp = np.abs(y_w_tr_phys).reshape(-1) > 0.15     # > 8.6 deg/s (roundabouts, sharp turns)
    is_accel = np.abs(y_dv_tr_phys).reshape(-1) > 0.15    # > 1.5 m/s^2 forward or brake

    turn_idx = np.where(is_turn)[0]
    sharp_idx = np.where(is_sharp)[0]
    accel_idx = np.where(is_accel)[0]

    print(f"[v6 Train] Rebalancing data for maneuver hyperfocus:")
    print(f"  Turn samples (|w| > 4.6 deg/s): {len(turn_idx):,}")
    print(f"  Sharp/Roundabout samples (|w| > 8.6 deg/s): {len(sharp_idx):,}")
    print(f"  High throttle/brake samples (|dv| > 1.5 m/s^2): {len(accel_idx):,}")

    # Replicate sharp/roundabout turns 3x, turns 1x, and high accel 1x
    rep_idx = np.concatenate([turn_idx, sharp_idx, sharp_idx, accel_idx])
    X_tr = np.concatenate([X_tr, X_tr[rep_idx]], axis=0)
    y_dv_tr = np.concatenate([y_dv_tr, y_dv_tr[rep_idx]], axis=0)
    y_w_tr = np.concatenate([y_w_tr, y_w_tr[rep_idx]], axis=0)
    y_z_tr = np.concatenate([y_z_tr, y_z_tr[rep_idx]], axis=0)
    y_ba_tr = np.concatenate([y_ba_tr, y_ba_tr[rep_idx]], axis=0)
    y_bw_tr = np.concatenate([y_bw_tr, y_bw_tr[rep_idx]], axis=0)
    weights_tr = np.concatenate([weights_tr, weights_tr[rep_idx]], axis=0)
    print(f"[v6 Train] Total training windows after maneuver rebalancing: {X_tr.shape[0]:,}")

    # ----------------------------------------------------------
    # Datasets & loaders
    # ----------------------------------------------------------
    ds_train = V6AugmentedDataset(
        X_tr, y_dv_tr, y_w_tr, y_z_tr, y_ba_tr, y_bw_tr,
        scaler_X=scalers["X"],
        sample_weights=weights_tr,
        is_train=use_augmentation,
        augment_prob=CONFIG["training"].get("augment_prob", 0.7),
    )
    ds_val = V6AugmentedDataset(
        X_va, y_dv_va, y_w_va, y_z_va, y_ba_va, y_bw_va,
        scaler_X=scalers["X"],
        sample_weights=np.ones(len(X_va), dtype=np.float32),
        is_train=False,
    )

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v6 Train] Device: {device}")

    model = PINODeadReckoningNetV6(
        in_channels=CONFIG["model"]["in_channels"],
        conv_channels=CONFIG["model"]["conv_channels"],
        gru_hidden=CONFIG["model"]["gru_hidden"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v6 Train] Trainable parameters: {total_params:,}")
    if total_params > 25000:
        raise RuntimeError(f"Parameter budget exceeded: {total_params:,} > 25,000")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5,
    )

    loss_weights = CONFIG["training"]["loss_weights"]

    # ----------------------------------------------------------
    # Early stopping on physical combined MAE
    # ----------------------------------------------------------
    best_score = float("inf")
    patience = int(CONFIG["training"]["patience"])
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "vel_mae": [],
        "head_mae": [],
        "combined_score": [],
        "lr": [],
    }

    print("\n" + "=" * 100)
    print(
        f"{'Epoch':<8}{'TrainLoss':<11}{'VelMAE':<10}{'HeadMAE':<10}"
        f"{'CombScore':<11}{'LR':<11}{'Time'}"
    )
    print("=" * 100)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()

        epoch_base_lr = optimizer.param_groups[0]["lr"]
        train_loss_values = []
        train_vel_values = []
        train_head_values = []

        for batch in loader_train:
            bx, b_ydv, b_yw, b_yz, b_yba, b_ybw, b_wgt = [
                value.to(device) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            preds = model(bx)
            targets = (b_ydv, b_yw, b_yz, b_yba, b_ybw)
            loss, metrics = compute_v6_loss(
                preds, targets, b_wgt, loss_weights,
                w_center=w_center, w_scale=w_scale,
                dv_center=dv_center, dv_scale=dv_scale,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()

            # ---- Batch-wise adaptive learning rate ----
            if use_batch_lr:
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0))
                target_gnorm = 1.0
                lr_scale = min(3.0, max(0.3, target_gnorm / (gnorm + 1e-6)))
                for pg in optimizer.param_groups:
                    pg["lr"] = epoch_base_lr * lr_scale
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_values.append(metrics["l_total"])
            train_vel_values.append(metrics["l_vel"])
            train_head_values.append(metrics["l_head"])

        if not use_batch_lr:
            # Restore the schedule LR after the epoch if not using batch-wise.
            scheduler.step()

        # ---- Validation ----
        vel_mae, head_mae, combined_score = compute_physical_mae(
            model, loader_val, scalers["y_dv"], scalers["y_w"], device,
        )

        mean_train_loss = float(np.mean(train_loss_values))
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        if use_batch_lr:
            scheduler.step()

        is_best = combined_score < best_score
        if is_best:
            best_score = combined_score
            patience_counter = 0
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_combined_mae": best_score,
                "best_vel_mae": vel_mae,
                "best_head_mae": head_mae,
                "config": CONFIG["model"],
                "training": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "subsample": subsample,
                    "use_smote": use_smote,
                    "use_augmentation": use_augmentation,
                    "use_batch_lr": use_batch_lr,
                    "loss_weights": loss_weights,
                },
            }
            torch.save(best_ckpt, CKPT_DIR / "best_model_v6.pth")
        else:
            patience_counter += 1

        history["epoch"].append(epoch)
        history["train_loss"].append(mean_train_loss)
        history["vel_mae"].append(vel_mae)
        history["head_mae"].append(head_mae)
        history["combined_score"].append(combined_score)
        history["lr"].append(current_lr)

        print(
            f"{epoch:<8}{mean_train_loss:<11.5f}{vel_mae:<10.4f}"
            f"{head_mae:<10.4f}{combined_score:<11.4f}"
            f"{current_lr:<11.2e}{epoch_time:.1f}s"
            + (" [BEST]" if is_best else f" ({patience_counter}/{patience})")
        )

        if patience_counter >= patience:
            print("\n[v6 Train] Early stopping.")
            break

    # Save training history
    with open(RESULTS_DIR / "training_history_v6.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[v6 Train] History saved to: {RESULTS_DIR / 'training_history_v6.json'}")

    print("\n[v6 Train] Training complete.")

    # Reload best checkpoint
    best_path = CKPT_DIR / "best_model_v6.pth"
    best_weights = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_weights["model_state_dict"])
    model.eval()
    print(f"[v6 Train] Best epoch: {best_weights['epoch']}, combined MAE: {best_score:.4f}")

    export_onnx_v6(model, str(CKPT_DIR / "best_model_v6.onnx"), device)
    export_torchscript_v6(model, str(CKPT_DIR / "best_model_v6_torchscript.pt"), device)
    print("[v6 Train] Exports complete.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINO-DR v6 training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE upscaling")
    parser.add_argument("--no-augment", action="store_true", help="Disable IMU augmentation")
    parser.add_argument("--use-batch-lr", action="store_true", help="Enable batch-wise LR scaling")
    args = parser.parse_args()

    train_v6(
        max_epochs=args.epochs or int(CONFIG["training"]["max_epochs"]),
        batch_size=args.batch_size or int(CONFIG["training"]["batch_size"]),
        lr=args.lr or float(CONFIG["training"]["learning_rate"]),
        subsample=args.subsample,
        use_smote=args.use_smote,
        use_augmentation=not args.no_augment,
        use_batch_lr=args.use_batch_lr,
    )
