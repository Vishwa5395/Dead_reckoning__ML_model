"""
preprocess_v2.py
----------------
Production-grade data cleaning for the cached IO-VNBD IDNN splits.

WHY THIS EXISTS
---------------
The original cached `data/preprocessed/dataset_splits.npz` was scaled with a
MinMaxScaler that fit a catastrophic GPS outlier: the displacement target has a
99.99th percentile of ~36 m/s, but a **maximum of ~1843.7 m/s** (a single GPS
jump). Because MinMaxScaler mapped [0, max] -> [0, 1], every realistic driving
sample (median ~10.8 m/s) compressed to ~0.006, collapsing the displacement
target distribution. The old model therefore learned an essentially constant
"~12 m/s" predictor, which is why it under-performs the pure INS baseline on
motorway and quick-acceleration scenarios.

Because the raw CSV dataset lives on another machine and is unavailable here,
this module reconstructs the raw values from the *existing, exactly invertible*
MinMaxScalers, removes physically-impossible outliers using thresholds derived
from the train split only, and re-fits a fresh, clean scaler.

No temporal leakage: thresholds are fixed physical constants (road-vehicle
limits), and scalers are fitted exclusively on the 70% training split.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_CACHE = ROOT / "data" / "preprocessed"
DST_CACHE = ROOT / "data" / "preprocessed" / "clean"

# Physical clip thresholds (road vehicle limits, not data-derived, so no leakage)
CLIP = {
    "disp_accel": (-8.0, 8.0),      # m/s^2  (real p99.99 ~ 6.25)
    "disp_disp": (0.0, 45.0),       # m      (real p99.99 ~ 36.4)
    "ori_gyro": (-1.0, 1.0),        # rad/s  (real p99.99 ~ 0.59)
    "ori_yaw": (-1.2, 1.2),         # rad/s  (real p99.99 ~ 0.71)
}


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def load_raw_splits() -> dict:
    """Reconstruct raw (unscaled) feature/target arrays for all splits."""
    src = SRC_CACHE / "dataset_splits.npz"
    if not src.exists():
        raise FileNotFoundError(f"Missing cached splits: {src}")
    d = np.load(src)
    with open(SRC_CACHE / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)

    out = {}
    for split in ["tr", "va", "te"]:
        # Displacement
        dX = sc["disp_X"].inverse_transform(d[f"d_X_{split}"])
        dY = sc["disp_Y"].inverse_transform(d[f"d_Y_{split}"])
        # Orientation
        oX = sc["ori_X"].inverse_transform(d[f"o_X_{split}"])
        oY = sc["ori_Y"].inverse_transform(d[f"o_Y_{split}"])
        out[split] = {"d_X": dX, "d_Y": dY, "o_X": oX, "o_Y": oY}
        print(
            f"[v2] Raw {split}: d_X={dX.shape} d_Y={dY.shape} "
            f"o_X={oX.shape} o_Y={oY.shape}"
        )
    return out


def clean_splits(raw: dict) -> dict:
    """Clip outliers and re-fit scalers on train only. Returns clean arrays."""
    print("\n[v2] Clipping outliers with physical thresholds:")
    for k, v in CLIP.items():
        print(f"       {k}: [{v[0]}, {v[1]}]")

    clean = {}
    for split in raw.keys():
        r = raw[split]
        c = {
            "d_X": np.concatenate(
                [
                    _clip(r["d_X"][:, :10], *CLIP["disp_accel"]),
                    _clip(r["d_X"][:, 10:], *CLIP["disp_disp"]),
                ],
                axis=1,
            ),
            "d_Y": _clip(r["d_Y"], *CLIP["disp_disp"]),
            "o_X": _clip(r["o_X"], *CLIP["ori_gyro"]),
            "o_Y": _clip(r["o_Y"], *CLIP["ori_yaw"]),
        }
        clean[split] = c

    # --- Re-fit scalers strictly on train ---
    tr = clean["tr"]
    scalers = {
        "disp_X": MinMaxScaler(feature_range=(0, 1)).fit(tr["d_X"]),
        "disp_Y": MinMaxScaler(feature_range=(0, 1)).fit(tr["d_Y"]),
        "ori_X": MinMaxScaler(feature_range=(0, 1)).fit(tr["o_X"]),
        "ori_Y": MinMaxScaler(feature_range=(0, 1)).fit(tr["o_Y"]),
    }

    # --- Transform all splits with the train-fitted scalers ---
    scaled = {}
    for split, c in clean.items():
        scaled[split] = {
            "d_X": scalers["disp_X"].transform(c["d_X"]).astype(np.float32),
            "d_Y": scalers["disp_Y"].transform(c["d_Y"]).astype(np.float32),
            "o_X": scalers["ori_X"].transform(c["o_X"]).astype(np.float32),
            "o_Y": scalers["ori_Y"].transform(c["o_Y"]).astype(np.float32),
        }

    for k in ["disp_X", "disp_Y", "ori_X", "ori_Y"]:
        mm = scalers[k]
        print(
            f"[v2] Clean scaler {k}: min_={np.round(mm.data_min_, 3)} "
            f"max_={np.round(mm.data_max_, 3)}"
        )

    return scaled, scalers


def save_clean(scaled: dict, scalers: dict) -> None:
    DST_CACHE.mkdir(parents=True, exist_ok=True)
    npz_path = DST_CACHE / "dataset_splits_clean.npz"
    np.savez_compressed(
        npz_path,
        d_X_tr=scaled["tr"]["d_X"], d_Y_tr=scaled["tr"]["d_Y"],
        o_X_tr=scaled["tr"]["o_X"], o_Y_tr=scaled["tr"]["o_Y"],
        d_X_va=scaled["va"]["d_X"], d_Y_va=scaled["va"]["d_Y"],
        o_X_va=scaled["va"]["o_X"], o_Y_va=scaled["va"]["o_Y"],
        d_X_te=scaled["te"]["d_X"], d_Y_te=scaled["te"]["d_Y"],
        o_X_te=scaled["te"]["o_X"], o_Y_te=scaled["te"]["o_Y"],
    )
    with open(DST_CACHE / "scalers_clean.pkl", "wb") as f:
        pickle.dump(scalers, f)

    meta = {
        "clip": CLIP,
        "train_samples": int(scaled["tr"]["d_X"].shape[0]),
        "val_samples": int(scaled["va"]["d_X"].shape[0]),
        "test_samples": int(scaled["te"]["d_X"].shape[0]),
        "note": (
            "Raw values reconstructed from original MinMaxScalers, clipped to "
            "physical road-vehicle limits, and re-scaled with fresh MinMaxScalers "
            "fitted on the 70% train split only."
        ),
    }
    with open(DST_CACHE / "clip_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[v2] Clean cache written to {DST_CACHE}")
    print(f"     {npz_path.name} ({npz_path.stat().st_size/1024:.1f} KB)")
    print(f"     scalers_clean.pkl ({ (DST_CACHE/'scalers_clean.pkl').stat().st_size/1024:.1f} KB)")


def main() -> None:
    raw = load_raw_splits()
    scaled, scalers = clean_splits(raw)
    save_clean(scaled, scalers)
    print("\n[v2] Data cleaning complete.")


if __name__ == "__main__":
    main()
