"""
split_data_five.py
------------------
Partitions the full dataset (72,614 training windows, 10,676 validation windows)
into 5 dedicated regime datasets for the 5 Supreme Specialists:
  1. Motorway (High-speed cruising, low yaw)
  2. Roundabout (Sustained circular cornering, high centripetal acceleration)
  3. Quick Accel (High positive forward acceleration)
  4. Hard Brake (High negative forward deceleration, stop transients)
  5. Sharp Turns (Transient high yaw rate, urban corners)

Each dataset includes 85% primary regime windows + 15% anchor windows
from opposing regimes to maintain cross-domain calibration and stability.
"""

import os
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_NPZ = ROOT.parent / "v4_turn_focused" / "data" / "dataset_splits_v4.npz"
SRC_SCALERS = ROOT.parent / "v4_turn_focused" / "data" / "scalers_v4.pkl"


def build_five_splits():
    print("[v7][5-Splits] Loading full v4 dataset...")
    d4 = np.load(SRC_NPZ)
    with open(SRC_SCALERS, "rb") as f:
        scalers = pickle.load(f)
    s_X = scalers["X"]

    X_tr = d4["X_tr"]      # (72614, 10, 6)
    yd_tr = d4["y_d_tr"]    # (72614, 1)
    yo_tr = d4["y_o_tr"]    # (72614, 1)
    yz_tr = d4["y_z_tr"]    # (72614, 1)

    X_va = d4["X_va"]      # (10676, 10, 6)
    yd_va = d4["y_d_va"]    # (10676, 1)
    yo_va = d4["y_o_va"]    # (10676, 1)
    yz_va = d4["y_z_va"]    # (10676, 1)

    # Invert X_tr to physical units for physical rule masking
    X_tr_phys = s_X.inverse_transform(X_tr.reshape(X_tr.shape[0], -1)).reshape(X_tr.shape[0], 10, 6)
    X_va_phys = s_X.inverse_transform(X_va.reshape(X_va.shape[0], -1)).reshape(X_va.shape[0], 10, 6)

    def compute_masks(X_p):
        mean_afwd = np.mean(X_p[:, :, 0], axis=1)
        mean_wyaw = np.mean(np.abs(X_p[:, :, 1]), axis=1)
        mean_alat = np.mean(np.abs(X_p[:, :, 2]), axis=1)
        v_last = X_p[:, -1, 3]

        # Masks for the 5 regimes
        m_roundabout = (mean_alat >= 1.5) & (v_last >= 4.0) & (v_last <= 18.0)
        m_hard_brake = (mean_afwd <= -0.8) | (X_p[:, -1, 0] <= -1.2)
        m_quick_accel = (mean_afwd >= 0.8) | (X_p[:, -1, 0] >= 1.2)
        m_sharp_turns = (mean_wyaw >= 0.04) & (~m_roundabout)
        m_motorway = (v_last >= 16.0) & (mean_wyaw < 0.035) & (mean_alat < 1.2) & (~m_hard_brake) & (~m_quick_accel)

        return {
            "roundabout": m_roundabout,
            "hard_brake": m_hard_brake,
            "quick_accel": m_quick_accel,
            "sharp_turns": m_sharp_turns,
            "motorway": m_motorway,
        }

    masks_tr = compute_masks(X_tr_phys)
    masks_va = compute_masks(X_va_phys)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    scenarios = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]

    print("\n[v7][5-Splits] Partitioning Statistics:")
    for scen in scenarios:
        m_t = masks_tr[scen]
        m_v = masks_va[scen]
        idx_primary = np.where(m_t)[0]
        idx_other = np.where(~m_t)[0]

        n_primary = len(idx_primary)
        n_anchor = int(n_primary * 0.15)
        anchor_sub = rng.choice(idx_other, size=min(n_anchor, len(idx_other)), replace=False)
        all_tr_idx = np.concatenate([idx_primary, anchor_sub])
        rng.shuffle(all_tr_idx)

        idx_va = np.where(m_v)[0]
        if len(idx_va) == 0:
            idx_va = rng.choice(len(X_va), size=500, replace=False)

        out_path = DATA_DIR / f"data_{scen}.npz"
        np.savez_compressed(
            out_path,
            X_tr=X_tr[all_tr_idx],
            yd_tr=yd_tr[all_tr_idx],
            yo_tr=yo_tr[all_tr_idx],
            yz_tr=yz_tr[all_tr_idx],
            X_va=X_va[idx_va],
            yd_va=yd_va[idx_va],
            yo_va=yo_va[idx_va],
            yz_va=yz_va[idx_va],
        )

        print(f"  - {scen:12s}: {n_primary:6d} primary + {len(anchor_sub):5d} anchors = {len(all_tr_idx):6d} train | {len(idx_va):5d} val -> {out_path.name}")

    print("[v7][5-Splits] All 5 scenario datasets partitioned and saved successfully.")


if __name__ == "__main__":
    build_five_splits()
