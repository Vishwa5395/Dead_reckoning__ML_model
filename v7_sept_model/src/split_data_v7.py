"""
split_data_v7.py
----------------
Data splitting pipeline for Dual-Specialist PINO-DR v7.

Spec implementation:
1. Load every training window pooled from across all sessions (72,614 windows).
2. Compute |yaw_rate| for each window: mean absolute physical yaw rate over the 10 timesteps.
3. Compute the 60th percentile threshold tau_60.
4. Straight-model training set (S1) = all windows below the 60th percentile (< tau_60).
5. Turn-model training set (S2) = all windows at or above the 60th percentile (>= tau_60).
6. Held-out validation slices:
   - S1 early stopping: low-yaw validation slice (|yaw_rate| < tau_60).
   - S2 early stopping: high-yaw validation slice (|yaw_rate| >= tau_60).
7. Saves:
   - data/dataset_splits_v7.npz
   - data/split_metadata_v7.json
   - data/scalers_v7.pkl
   - data/test_scenarios_v7.pkl
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
SRC_DATA_DIR = WS_ROOT / "v4_turn_focused" / "data"


def split_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src_npz = SRC_DATA_DIR / "dataset_splits_v4.npz"
    src_scalers = SRC_DATA_DIR / "scalers_v4.pkl"
    src_test_scenarios = SRC_DATA_DIR / "test_scenarios_v4.pkl"

    if not src_npz.exists() or not src_scalers.exists() or not src_test_scenarios.exists():
        raise FileNotFoundError(f"Source v4 data files missing in {SRC_DATA_DIR}")

    print("[v7][Split] Loading pooled training windows from v4 preprocessed set...")
    raw = np.load(src_npz)
    X_tr = raw["X_tr"]
    y_d_tr = raw["y_d_tr"]
    y_o_tr = raw["y_o_tr"]
    y_z_tr = raw["y_z_tr"]

    X_va = raw["X_va"]
    y_d_va = raw["y_d_va"]
    y_o_va = raw["y_o_va"]
    y_z_va = raw["y_z_va"]

    X_te = raw["X_te"]
    y_d_te = raw["y_d_te"]
    y_o_te = raw["y_o_te"]
    y_z_te = raw["y_z_te"]

    n_tr = len(X_tr)
    n_va = len(X_va)
    n_te = len(X_te)
    print(f"[v7][Split] Total windows: Train={n_tr:,}, Val={n_va:,}, Test={n_te:,}")

    # Compute physical yaw rate across timesteps (channel 1: scaled in [0, 1] where 0.5 is 0 rad/s)
    w_phys_tr = X_tr[:, :, 1] * 2.0 - 1.0
    w_phys_va = X_va[:, :, 1] * 2.0 - 1.0

    # Mean absolute yaw rate across the 10 timesteps of each window
    mean_abs_w_tr = np.mean(np.abs(w_phys_tr), axis=1)
    mean_abs_w_va = np.mean(np.abs(w_phys_va), axis=1)

    # 60th percentile threshold strictly on train split
    tau_60 = float(np.percentile(mean_abs_w_tr, 60))
    tau_60_deg = float(np.degrees(tau_60))

    print("\n" + "=" * 70)
    print(f"[v7][Split] 60th PERCENTILE YAW-RATE THRESHOLD: {tau_60:.5f} rad/s ({tau_60_deg:.2f} deg/s)")
    print("=" * 70)

    # Masks for S1 (< tau_60) and S2 (>= tau_60)
    mask_s1_tr = mean_abs_w_tr < tau_60
    mask_s2_tr = mean_abs_w_tr >= tau_60

    mask_s1_va = mean_abs_w_va < tau_60
    mask_s2_va = mean_abs_w_va >= tau_60

    # S1 Straight Specialist: 4 channels [a_fwd, w_yaw, a_lat, v_prev]
    X_tr_s1 = X_tr[mask_s1_tr][:, :, :4]
    y_d_tr_s1 = y_d_tr[mask_s1_tr]
    y_o_tr_s1 = y_o_tr[mask_s1_tr]
    y_z_tr_s1 = y_z_tr[mask_s1_tr]

    X_va_s1 = X_va[mask_s1_va][:, :, :4]
    y_d_va_s1 = y_d_va[mask_s1_va]
    y_o_va_s1 = y_o_va[mask_s1_va]
    y_z_va_s1 = y_z_va[mask_s1_va]

    # S2 Turning Specialist: all 6 channels [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_residual]
    X_tr_s2 = X_tr[mask_s2_tr]
    y_d_tr_s2 = y_d_tr[mask_s2_tr]
    y_o_tr_s2 = y_o_tr[mask_s2_tr]
    y_z_tr_s2 = y_z_tr[mask_s2_tr]

    X_va_s2 = X_va[mask_s2_va]
    y_d_va_s2 = y_d_va[mask_s2_va]
    y_o_va_s2 = y_o_va[mask_s2_va]
    y_z_va_s2 = y_z_va[mask_s2_va]

    # ── Anchor Replay Augmentation (85% Target Regime + 15% Cross-Regime Anchors) ──
    # Prevents catastrophic specialization where fine-tuning destroys baseline calibration
    np.random.seed(42)
    n_replay_s1 = int(len(X_tr_s1) * 0.15)
    n_replay_s2 = int(len(X_tr_s2) * 0.15)

    idx_s2_for_s1 = np.random.choice(len(X_tr_s2), size=n_replay_s1, replace=False)
    idx_s1_for_s2 = np.random.choice(len(X_tr_s1), size=n_replay_s2, replace=False)

    X_tr_s1_replay = np.concatenate([X_tr_s1, X_tr_s2[idx_s2_for_s1][:, :, :4]], axis=0)
    y_d_tr_s1_replay = np.concatenate([y_d_tr_s1, y_d_tr_s2[idx_s2_for_s1]], axis=0)
    y_o_tr_s1_replay = np.concatenate([y_o_tr_s1, y_o_tr_s2[idx_s2_for_s1]], axis=0)
    y_z_tr_s1_replay = np.concatenate([y_z_tr_s1, y_z_tr_s2[idx_s2_for_s1]], axis=0)

    # For S2 replay: use full 6-channel real data from straight driving windows
    s1_replay_6ch = X_tr[mask_s1_tr][idx_s1_for_s2]

    X_tr_s2_replay = np.concatenate([X_tr_s2, s1_replay_6ch], axis=0)
    y_d_tr_s2_replay = np.concatenate([y_d_tr_s2, y_d_tr_s1[idx_s1_for_s2]], axis=0)
    y_o_tr_s2_replay = np.concatenate([y_o_tr_s2, y_o_tr_s1[idx_s1_for_s2]], axis=0)
    y_z_tr_s2_replay = np.concatenate([y_z_tr_s2, y_z_tr_s1[idx_s1_for_s2]], axis=0)

    print(f"  S1 Anchor Replay Pool:         {len(X_tr_s1_replay):,} windows (43,568 target + {n_replay_s1:,} anchors)")
    print(f"  S2 Anchor Replay Pool:         {len(X_tr_s2_replay):,} windows (29,046 target + {n_replay_s2:,} anchors)")
    print("=" * 70)

    # Save partitioned npz
    out_npz = DATA_DIR / "dataset_splits_v7.npz"
    np.savez_compressed(
        out_npz,
        X_tr_s1=X_tr_s1, y_d_tr_s1=y_d_tr_s1, y_o_tr_s1=y_o_tr_s1, y_z_tr_s1=y_z_tr_s1,
        X_va_s1=X_va_s1, y_d_va_s1=y_d_va_s1, y_o_va_s1=y_o_va_s1, y_z_va_s1=y_z_va_s1,
        X_tr_s2=X_tr_s2, y_d_tr_s2=y_d_tr_s2, y_o_tr_s2=y_o_tr_s2, y_z_tr_s2=y_z_tr_s2,
        X_va_s2=X_va_s2, y_d_va_s2=y_d_va_s2, y_o_va_s2=y_o_va_s2, y_z_va_s2=y_z_va_s2,
        X_tr_s1_replay=X_tr_s1_replay, y_d_tr_s1_replay=y_d_tr_s1_replay,
        y_o_tr_s1_replay=y_o_tr_s1_replay, y_z_tr_s1_replay=y_z_tr_s1_replay,
        X_tr_s2_replay=X_tr_s2_replay, y_d_tr_s2_replay=y_d_tr_s2_replay,
        y_o_tr_s2_replay=y_o_tr_s2_replay, y_z_tr_s2_replay=y_z_tr_s2_replay,
        X_te=X_te, y_d_te=y_d_te, y_o_te=y_o_te, y_z_te=y_z_te,
    )
    print(f"[v7][Split] Saved partitioned splits -> {out_npz}")

    # Copy scalers and test scenarios
    shutil.copyfile(src_scalers, DATA_DIR / "scalers_v7.pkl")
    shutil.copyfile(src_test_scenarios, DATA_DIR / "test_scenarios_v7.pkl")
    print(f"[v7][Split] Synced scalers and test scenarios -> {DATA_DIR}")

    # Save metadata
    metadata = {
        "version": "v7_sept_model",
        "description": "Dual-specialist regime partitioning at 60th percentile of |yaw_rate|",
        "tau_60_rad_s": tau_60,
        "tau_60_deg_s": tau_60_deg,
        "s1_straight_specialist": {
            "initialization": "v3_pino_dr/checkpoints/best_model.pth",
            "in_channels": 4,
            "channels": ["a_fwd", "w_yaw", "a_lat", "v_prev"],
            "train_samples": int(len(X_tr_s1)),
            "train_pct": float(len(X_tr_s1) / n_tr * 100),
            "val_held_out_samples": int(len(X_va_s1)),
            "val_pct": float(len(X_va_s1) / n_va * 100),
        },
        "s2_turning_specialist": {
            "initialization": "v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth",
            "in_channels": 6,
            "channels": ["a_fwd", "w_yaw", "a_lat", "v_prev", "w_yaw_accel", "centripetal_residual"],
            "train_samples": int(len(X_tr_s2)),
            "train_pct": float(len(X_tr_s2) / n_tr * 100),
            "val_held_out_samples": int(len(X_va_s2)),
            "val_pct": float(len(X_va_s2) / n_va * 100),
        },
        "test_samples": int(n_te),
    }
    out_meta = DATA_DIR / "split_metadata_v7.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[v7][Split] Saved metadata -> {out_meta}")


if __name__ == "__main__":
    split_data()
