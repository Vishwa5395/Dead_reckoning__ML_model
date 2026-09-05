"""
train_v2.py
-----------
Production-grade training for the v2 Displacement + Orientation IDNN models.

Key improvements over the legacy train.py:
- Deterministic reproducibility (seeds for torch, numpy, python, cudnn).
- AdamW optimiser with weight decay (regularisation).
- ReduceLROnPlateau scheduler (adaptive LR decay on plateaus).
- Early stopping on validation MAE (patience) + best-checkpoint restoration.
- Gradient clipping (max-norm) for training stability.
- Full per-epoch Train/Val MAE history persisted to results for reporting.
- Clean scaler pipeline (clipped, re-fitted) loaded from data/preprocessed/clean.
- Saves .pth (with metadata), ONNX and TorchScript exports under checkpoints_v2/.

Config is passed entirely via a single CONFIG dict and CLI overrides, so the
script is deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models_v2 import DisplacementIDNNV2, OrientationIDNNV2, export_to_onnx_v2

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "displacement": {
        "epochs": 120,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "grad_clip": 1.0,
        "dropout": 0.20,
        "hidden_dims": (64, 64, 32),
    },
    "orientation": {
        "epochs": 120,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "grad_clip": 1.0,
        "dropout": 0.20,
        "hidden_dims": (64, 64, 32),
    },
}

# Fixed physical bounds used at inference to keep feedback in valid range
DISP_BOUNDS = (0.0, 45.0)
ORI_BOUNDS = (-1.2, 1.2)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clean_data(cache_dir: Path):
    """Load the cleaned, re-scaled splits and scalers."""
    npz = cache_dir / "dataset_splits_clean.npz"
    pkl = cache_dir / "scalers_clean.pkl"
    if not npz.exists() or not pkl.exists():
        raise FileNotFoundError(
            f"Clean cache missing. Run `python src/preprocess_v2.py` first. "
            f"Need {npz} and {pkl}"
        )
    d = np.load(npz)
    with open(pkl, "rb") as f:
        import pickle
        scalers = pickle.load(f)
    return d, scalers


def build_loaders(d, model_type: str, batch_size: int, device: torch.device):
    if model_type == "displacement":
        X_tr = torch.tensor(d["d_X_tr"], dtype=torch.float32).to(device)
        Y_tr = torch.tensor(d["d_Y_tr"], dtype=torch.float32).to(device)
        X_va = torch.tensor(d["d_X_va"], dtype=torch.float32).to(device)
        Y_va = torch.tensor(d["d_Y_va"], dtype=torch.float32).to(device)
        X_te = torch.tensor(d["d_X_te"], dtype=torch.float32).to(device)
        Y_te = torch.tensor(d["d_Y_te"], dtype=torch.float32).to(device)
    elif model_type == "orientation":
        X_tr = torch.tensor(d["o_X_tr"], dtype=torch.float32).to(device)
        Y_tr = torch.tensor(d["o_Y_tr"], dtype=torch.float32).to(device)
        X_va = torch.tensor(d["o_X_va"], dtype=torch.float32).to(device)
        Y_va = torch.tensor(d["o_Y_va"], dtype=torch.float32).to(device)
        X_te = torch.tensor(d["o_X_te"], dtype=torch.float32).to(device)
        Y_te = torch.tensor(d["o_Y_te"], dtype=torch.float32).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    train_loader = DataLoader(
        TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_va, Y_va), batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader, (X_tr.shape[0], X_va.shape[0], X_te.shape[0])


def make_loader_sizes(model_type: str):
    if model_type == "displacement":
        return 20
    return 10


@torch.no_grad()
def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        pred = model(x)
        total += (pred - y).abs().sum().item()
        n += y.shape[0]
    return total / n


def train_model(
    model_type: str,
    config: dict,
    cache_dir: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    device: torch.device,
    seed: int = 42,
):
    set_seed(seed)
    d, scalers = load_clean_data(cache_dir)
    train_loader, val_loader, test_loader, sizes = build_loaders(
        d, model_type, config["batch_size"], device
    )
    n_train, n_val, n_test = sizes

    input_dim = make_loader_sizes(model_type)

    if model_type == "displacement":
        model = DisplacementIDNNV2(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        ).to(device)
    else:
        model = OrientationIDNNV2(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        ).to(device)

    criterion = nn.L1Loss()  # MAE per paper
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*78}")
    print(f"TRAINING v2 {model_type.upper()} IDNN")
    print(f"{'='*78}")
    print(f"Samples: Train={n_train:,} Val={n_val:,} Test={n_test:,}")
    print(f"Architecture: {input_dim} -> {config['hidden_dims']} -> 1  ({n_params:,} params)")
    print(f"Optimizer: AdamW(lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"Loss: MAE | Batch size: {config['batch_size']} | Device: {device}")
    print(f"Early stop patience: {config['patience']} | Grad clip: {config['grad_clip']}")

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = checkpoint_dir / f"{model_type}_idnn.pth"
    history = []

    t0 = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * y.shape[0]
        train_mae = train_loss / n_train

        val_mae = evaluate_mae(model, val_loader, device)
        scheduler.step(val_mae)

        history.append(
            {"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae,
             "lr": optimizer.param_groups[0]["lr"]}
        )

        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "train_mae": train_mae,
                    "model_type": model_type,
                    "seed": seed,
                    "config": config,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        log = (
            f"Epoch [{epoch:3d}/{config['epochs']}] Train MAE: {train_mae:.5f} | "
            f"Val MAE: {val_mae:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            f"{' [*Best]' if improved else ''}"
        )
        if epoch % 5 == 0 or epoch == 1 or improved or epochs_no_improve == 1:
            print(log)

        if epochs_no_improve >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improve for {config['patience']} epochs).")
            break

    total_time = time.time() - t0

    # Restore best and evaluate on held-out test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_mae = evaluate_mae(model, test_loader, device)

    print(f"\n>>> v2 {model_type.upper()} HELD-OUT TEST MAE: {test_mae:.5f} <<<")
    print(f"Best epoch: {best_epoch} | Best Val MAE: {best_val:.5f} | Training time: {total_time:.1f}s")

    # Export ONNX / TorchScript
    model.eval()
    dummy = torch.randn(1, input_dim, device=device)
    onnx_path = checkpoint_dir / f"{model_type}_idnn.onnx"
    export_to_onnx_v2(
        model, dummy, str(onnx_path),
        input_name="imu_input", output_name="prediction",
    )

    # Save history
    history_path = results_dir / f"training_history_{model_type}.json"
    with open(history_path, "w") as f:
        json.dump(
            {"model_type": model_type, "history": history,
             "best_epoch": best_epoch, "best_val_mae": best_val,
             "test_mae": test_mae, "total_time_s": total_time},
            f, indent=2,
        )
    print(f"[v2] Training history saved: {history_path}")

    return {
        "model_type": model_type,
        "best_val_mae": best_val,
        "test_mae": test_mae,
        "best_epoch": best_epoch,
        "total_time_s": total_time,
        "n_params": n_params,
    }


def main(model_type: str = "all", seed: int = 42) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = ROOT / "data" / "preprocessed" / "clean"
    checkpoint_dir = ROOT / "checkpoints_v2"
    results_dir = ROOT / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = ["displacement", "orientation"] if model_type == "all" else [model_type]
    summary = {}
    for mt in models:
        summary[mt] = train_model(
            mt, CONFIG[mt], cache_dir, checkpoint_dir, results_dir, device, seed=seed
        )

    with open(results_dir / "training_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[v2] Training complete. Summary:")
    for mt, res in summary.items():
        print(f"  {mt}: test MAE={res['test_mae']:.5f} best_val={res['best_val_mae']:.5f} "
              f"best_epoch={res['best_epoch']} params={res['n_params']:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["displacement", "orientation", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(model_type=args.model, seed=args.seed)
