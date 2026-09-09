"""
train_moe_v7.py
---------------
Training & Assembly Engine for Supreme 5-Expert Neural Mixture-of-Experts (MoE).

Pipeline:
1. Loads 72,614 train & 10,676 val windows from the full dataset.
2. Instantiates SupremeMoENet containing 5 recurrent neural experts:
     - Expert 0: Motorway Cruising Specialist (4-ch) <- v3 weights (7.13m baseline)
     - Expert 1: Roundabout Specialist (6-ch) <- v4-D weights (55.37m baseline)
     - Expert 2: Quick Accel Specialist (6-ch) <- v4-D weights (19.58m baseline)
     - Expert 3: Hard Brake Specialist (4-ch) <- v3 weights (17.15m baseline)
     - Expert 4: Sharp Turns Specialist (6-ch) <- v4-D weights (36.39m baseline)
3. Trains the PhysicalNeuralRouter with continuous KL-Divergence on kinematic soft targets.
4. Validates on the 10,676 held-out validation windows.
5. Saves unified PyTorch checkpoint (best_supreme_moe.pth).
6. Exports TorchScript and ONNX models for mobile application integration.
"""

from __future__ import annotations

import os
import pickle
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
RESULTS_DIR = ROOT / "results"

SRC_NPZ = WS_ROOT / "v4_turn_focused" / "data" / "dataset_splits_v4.npz"
SRC_SCALERS = WS_ROOT / "v4_turn_focused" / "data" / "scalers_v4.pkl"
V3_CKPT = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
V4_D_CKPT = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"

from v7_sept_model.src.moe_five_model import (
    SupremeMoENet,
    export_onnx_moe,
    export_torchscript_moe,
)


def load_dataset(device: torch.device):
    """Loads 72,614 train & 10,676 val windows and computes kinematic soft targets."""
    print("[MoE][Data] Loading v4 dataset from:", SRC_NPZ.name)
    d = np.load(SRC_NPZ)
    with open(SRC_SCALERS, "rb") as f:
        scalers = pickle.load(f)
    s_X = scalers["X"]

    X_tr = d["X_tr"]  # (72614, 10, 6)
    X_va = d["X_va"]  # (10676, 10, 6)

    # Invert to physical units for physical kinematic targets
    X_tr_p = s_X.inverse_transform(X_tr.reshape(X_tr.shape[0], -1)).reshape(X_tr.shape[0], 10, 6)
    X_va_p = s_X.inverse_transform(X_va.reshape(X_va.shape[0], -1)).reshape(X_va.shape[0], 10, 6)

    def generate_soft_targets(X_p: np.ndarray) -> np.ndarray:
        v_last = X_p[:, -1, 3]
        alat_mean = np.mean(np.abs(X_p[:, :, 2]), axis=1)
        afwd_mean = np.mean(X_p[:, :, 0], axis=1)
        N = len(X_p)
        W = np.zeros((N, 5), dtype=np.float32)

        # Regimes:
        # 0: motorway (high speed >= 22.5 m/s, cruising afwd < 0.4)
        # 1: roundabout (sustained curvature >= 1.0 m/s^2 at 4..18 m/s)
        # 2: quick accel (afwd >= 0.5 m/s^2)
        # 3: hard brake (afwd <= -0.8 m/s^2, low lateral)
        # 4: sharp turns / general urban cornering (default)
        is_mot = (v_last >= 22.5) & (afwd_mean < 0.4)
        is_hb  = (afwd_mean <= -0.8) & (alat_mean < 0.8)
        is_rb  = (alat_mean >= 1.0) & (v_last >= 4.0) & (v_last <= 18.0)
        is_qa  = (afwd_mean >= 0.5)

        W[is_mot] = [0.98, 0.00, 0.00, 0.00, 0.02]
        W[is_hb & ~is_mot] = [0.00, 0.00, 0.05, 0.90, 0.05]
        W[is_rb & ~is_mot & ~is_hb] = [0.00, 0.90, 0.05, 0.00, 0.05]
        W[is_qa & ~is_mot & ~is_hb & ~is_rb] = [0.00, 0.01, 0.94, 0.01, 0.04]

        unassigned = ~is_mot & ~is_hb & ~is_rb & ~is_qa
        W[unassigned] = [0.00, 0.02, 0.02, 0.01, 0.95]
        return W

    W_tr = generate_soft_targets(X_tr_p)
    W_va = generate_soft_targets(X_va_p)

    print(f"[MoE][Data] Soft Target Distribution in Train (72,614 windows):")
    print(f"  Expert 0 (Motorway):    {np.sum(W_tr[:, 0] > 0.5):5d} ({np.mean(W_tr[:, 0] > 0.5)*100:.1f}%)")
    print(f"  Expert 1 (Roundabout):  {np.sum(W_tr[:, 1] > 0.5):5d} ({np.mean(W_tr[:, 1] > 0.5)*100:.1f}%)")
    print(f"  Expert 2 (Quick Accel): {np.sum(W_tr[:, 2] > 0.5):5d} ({np.mean(W_tr[:, 2] > 0.5)*100:.1f}%)")
    print(f"  Expert 3 (Hard Brake):  {np.sum(W_tr[:, 3] > 0.5):5d} ({np.mean(W_tr[:, 3] > 0.5)*100:.1f}%)")
    print(f"  Expert 4 (Sharp Turns): {np.sum(W_tr[:, 4] > 0.5):5d} ({np.mean(W_tr[:, 4] > 0.5)*100:.1f}%)")

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    t_W_tr = torch.tensor(W_tr, dtype=torch.float32, device=device)
    t_X_va = torch.tensor(X_va, dtype=torch.float32, device=device)
    t_W_va = torch.tensor(W_va, dtype=torch.float32, device=device)

    return (t_X_tr, t_W_tr), (t_X_va, t_W_va)


def train_neural_router(
    moe_model: SupremeMoENet,
    train_data,
    val_data,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
):
    print(f"\n{'='*75}\n[Training] Training Physical Neural Gating Router ({epochs} Epochs)\n{'='*75}")
    t_X_tr, t_W_tr = train_data
    t_X_va, t_W_va = val_data

    loader = DataLoader(
        TensorDataset(t_X_tr, t_W_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(moe_model.gating.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        moe_model.gating.train()
        tot_loss = 0.0
        nb = 0

        for xb, wb in loader:
            optimizer.zero_grad()
            feats = moe_model.gating.extract_features(xb)
            log_probs = torch.log_softmax(moe_model.gating.mlp(feats), dim=-1)
            loss = loss_fn(log_probs, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(moe_model.gating.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item()
            nb += 1

        scheduler.step()

        # Validation
        moe_model.gating.eval()
        with torch.no_grad():
            va_feats = moe_model.gating.extract_features(t_X_va)
            va_log_probs = torch.log_softmax(moe_model.gating.mlp(va_feats), dim=-1)
            va_loss = loss_fn(va_log_probs, t_W_va).item()
            va_pred = moe_model.gating(t_X_va).argmax(dim=-1)
            va_true = t_W_va.argmax(dim=-1)
            va_acc = (va_pred == va_true).float().mean().item()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in moe_model.gating.state_dict().items()}

        if ep % 3 == 0 or ep == epochs:
            print(f"[Router] Ep {ep:02d}/{epochs:02d} | Train KL Loss: {tot_loss/nb:.4f} | Val KL Loss: {va_loss:.4f} (best: {best_val_loss:.4f}) | Val Match Acc: {va_acc*100:.1f}%")

    moe_model.gating.load_state_dict(best_state)
    print(f"\n[Router] Gating Network trained in {time.time()-t0:.1f}s. Best Val KL Loss: {best_val_loss:.4f}")
    return moe_model


def run_moe_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"SUPREME 5-EXPERT MoE TRAINING & ASSEMBLY PIPELINE ON: {device}")
    print("=" * 80)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data, val_data = load_dataset(device)

    # 1. Instantiate SupremeMoENet
    moe_net = SupremeMoENet().to(device)

    # 2. Load the verified neural experts
    moe_net.load_expert_weights(
        v3_ckpt_path=V3_CKPT,
        v4_ckpt_path=V4_D_CKPT,
        device=device,
    )

    # 3. Train the Physical Neural Gating Router
    trained_moe = train_neural_router(
        moe_model=moe_net,
        train_data=train_data,
        val_data=val_data,
        epochs=15,
        batch_size=256,
        lr=1e-3,
        device=device,
    )

    # 4. Save PyTorch checkpoint
    out_pth = CKPT_DIR / "best_supreme_moe.pth"
    torch.save(
        {
            "model_state_dict": trained_moe.state_dict(),
            "num_experts": 5,
            "architecture": "SupremeMoENet",
            "expert_names": SupremeMoENet.EXPERT_NAMES,
        },
        out_pth,
    )
    print(f"\n[MoE] Saved Unified Checkpoint -> {out_pth.name}")

    # 5. Export ONNX & TorchScript
    onnx_path = CKPT_DIR / "best_supreme_moe.onnx"
    ts_path = CKPT_DIR / "best_supreme_moe_torchscript.pt"
    export_onnx_moe(trained_moe, onnx_path, device)
    export_torchscript_moe(trained_moe, ts_path, device)

    print("\n" + "=" * 80)
    print("SUPREME 5-EXPERT MoE TRAINING & ASSEMBLY COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return trained_moe


if __name__ == "__main__":
    run_moe_training_pipeline()
