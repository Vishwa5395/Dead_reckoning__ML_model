"""
preprocess_specialists.py
-------------------------
Rigorous Native 10 Hz Preprocessing Pipeline for Two Independent Specialist Models:
  - Model A (Turning Specialist): Roundabouts, Sharp Turns, High Lateral Dynamics
  - Model B (Longitudinal Specialist): Motorway, Acceleration, Braking, Standstill (ZUPT)

Step 1 Audit & Correctness Fixes:
  1. Sub-second cross-correlation synchronization between smartphone IMU and vehicle CAN.
  2. Gyroscope vertical turning axis identification & covariance sign alignment.
  3. Channel-wise feature standard scaling across all timesteps (shape N*T, 6).
  4. Zero-centered target scaling (with_mean=False) so straight/cruise motion maps to 0.0.
  5. Physical-unit IMU augmentation applied strictly to training splits; validation splits untouched.
  6. Strict frozen 50 Train / 9 Val / 13 Test split (zero journey overlap).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v6_smartphone_idr.src.augmentation_v6 import IMUAugmentor

DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "v6_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DATASET_BASE_DIR = CONFIG["dataset"]["base_dir"]
WINDOW_SIZE = CONFIG["window_size_10hz"]  # 20 steps @ 10 Hz = 2.0 seconds
DT = CONFIG["dt"]                          # 0.1 s
CLIP = CONFIG["clip_limits"]

TEST_SCENARIOS = {
    "motorway": ["Vw12"],
    "roundabout": ["Vta11"],
    "quick_accel": ["Vta12"],
    "hard_brake": ["Vw16b", "Vw17", "Vta9"],
    "sharp_turns": ["Vw6", "Vw7", "Vw8"],
}


def compute_vectorized_geodesic_disp(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized high-precision WGS-84 ellipsoidal geodesic displacement (m/step)."""
    n = len(lats)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    dlat = np.diff(lats) * (math.pi / 180.0)
    dlon = np.diff(lons) * (math.pi / 180.0)
    lat_mid = 0.5 * (lats[:-1] + lats[1:]) * (math.pi / 180.0)

    e2 = 0.00669437999014
    sin_lat = np.sin(lat_mid)
    denom = np.sqrt(1.0 - e2 * sin_lat**2)
    M = 6378137.0 * (1.0 - e2) / (denom**3)
    N = 6378137.0 / denom

    dx = N * np.cos(lat_mid) * dlon
    dy = M * dlat
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.where(dist > 15.0, 0.0, dist)  # Filter GPS glitch jumps (>540 km/h)

    disp = np.zeros(n, dtype=np.float64)
    disp[1:] = dist
    disp[0] = disp[1] if n > 1 else 0.0
    return disp


def find_synchronized_pairs(base_dir=DATASET_BASE_DIR):
    """Finds all synchronized (S-*.csv, V-*.csv) file pairs."""
    s_files, v_files = {}, {}
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue
            full = os.path.join(root, f)
            fl = f.lower()
            if fl.startswith("s-"):
                key = fl.replace("s-", "").replace(".csv", "").replace("-", "").replace("_", "")
                s_files[key] = full
            elif fl.startswith("v-"):
                key = fl.replace("v-", "").replace(".csv", "").replace("-", "").replace("_", "")
                v_files[key] = full
    paired = {}
    for key, s_p in s_files.items():
        if key in v_files:
            paired[key] = (s_p, v_files[key])
    return paired


def parse_journey_descriptions(pdf_path: Path) -> dict[str, str]:
    """Extracts ground-truth journey scenario descriptions from IO-VNBD README_1.pdf."""
    if not pdf_path.exists():
        return {}

    reader = pypdf.PdfReader(str(pdf_path))
    full_pdf = ""
    for page in reader.pages[4:15]:
        full_pdf += page.extract_text() + "\n"

    pattern = re.compile(r"V-(V[a-z0-9]+|S[0-9a-z]+|M|Y[0-9]*)", re.IGNORECASE)
    journey_desc = {}
    cur_j = None
    cur_text = []

    for line in full_pdf.split("\n"):
        m = pattern.search(line)
        if m:
            if cur_j:
                journey_desc[cur_j.lower()] = " ".join(cur_text)
            cur_j = m.group(1).lower().replace("-", "").replace("_", "")
            cur_text = [line]
        elif cur_j:
            cur_text.append(line)
    if cur_j:
        journey_desc[cur_j.lower()] = " ".join(cur_text)

    return journey_desc


def process_journey_specialists(s_path: str, v_path: str, name: str = "journey") -> dict | None:
    """
    Processes one synchronized journey pair natively at 10 Hz with audit corrections:
      1. Timestamp alignment and sub-second lag compensation.
      2. 3D gravity separation into horizontal acceleration and vertical gyro axis.
      3. Phone-to-vehicle longitudinal alignment via covariance with indicated acceleration.
      4. True vertical turning axis selection (col 16) with sign alignment.
    """
    try:
        df_s = pd.read_csv(s_path, encoding="latin1")
        df_v = pd.read_csv(v_path)
    except Exception:
        return None

    df_v.columns = [c.strip() for c in df_v.columns]
    min_len = min(len(df_s), len(df_v))
    if min_len < 60:
        return None

    df_s = df_s.iloc[:min_len]
    df_v = df_v.iloc[:min_len]

    # IMU raw readings
    ax = df_s.iloc[:, 9].values.astype(np.float64)
    ay = df_s.iloc[:, 10].values.astype(np.float64)
    az = df_s.iloc[:, 11].values.astype(np.float64)
    gx = df_s.iloc[:, 12].values.astype(np.float64)
    gy = df_s.iloc[:, 13].values.astype(np.float64)
    gz = df_s.iloc[:, 14].values.astype(np.float64)

    # Gyro columns: Col 15 (Yaw), Col 16 (Pitch), Col 17 (Roll)
    g15 = df_s.iloc[:, 15].values.astype(np.float64)
    g16 = df_s.iloc[:, 16].values.astype(np.float64)
    g17 = df_s.iloc[:, 17].values.astype(np.float64)

    # Vehicle reference dynamics
    v_acc = df_v["Indicated Longitudinal Acceleration (g)"].values.astype(np.float64) * 9.80665
    v_yaw_rate = np.radians(df_v["Yaw Rate (deg/sec)"].values.astype(np.float64))

    # 1. Sub-second lag synchronization between phone IMU and CAN vehicle reference
    # Test lags between -10 and +10 steps (-1.0s to +1.0s)
    best_lag = 0
    best_corr = -1.0
    if np.std(v_yaw_rate) > 0.03:
        target_ref = v_yaw_rate
        candidate_sig = g16
    else:
        target_ref = v_acc
        candidate_sig = ax

    m = len(target_ref)
    for lag in range(-10, 11):
        if lag >= 0:
            s_cand = candidate_sig[:m - lag]
            v_ref = target_ref[lag:]
        else:
            s_cand = candidate_sig[-lag:]
            v_ref = target_ref[:m + lag]
        if len(s_cand) > 30 and np.std(s_cand) > 1e-4 and np.std(v_ref) > 1e-4:
            r = abs(np.corrcoef(s_cand, v_ref)[0, 1])
            if not np.isnan(r) and r > best_corr:
                best_corr = r
                best_lag = lag

    # Apply lag shift
    if best_lag > 0:
        df_s = df_s.iloc[:min_len - best_lag].reset_index(drop=True)
        df_v = df_v.iloc[best_lag:min_len].reset_index(drop=True)
    elif best_lag < 0:
        df_s = df_s.iloc[-best_lag:min_len].reset_index(drop=True)
        df_v = df_v.iloc[:min_len + best_lag].reset_index(drop=True)

    n_aligned = min(len(df_s), len(df_v))
    if n_aligned < 50:
        return None

    df_s = df_s.iloc[:n_aligned]
    df_v = df_v.iloc[:n_aligned]

    ax = df_s.iloc[:, 9].values.astype(np.float64)
    ay = df_s.iloc[:, 10].values.astype(np.float64)
    az = df_s.iloc[:, 11].values.astype(np.float64)
    gx = df_s.iloc[:, 12].values.astype(np.float64)
    gy = df_s.iloc[:, 13].values.astype(np.float64)
    gz = df_s.iloc[:, 14].values.astype(np.float64)
    g15 = df_s.iloc[:, 15].values.astype(np.float64)
    g16 = df_s.iloc[:, 16].values.astype(np.float64)
    g17 = df_s.iloc[:, 17].values.astype(np.float64)

    v_acc = df_v["Indicated Longitudinal Acceleration (g)"].values.astype(np.float64) * 9.80665
    v_yaw_rate = np.radians(df_v["Yaw Rate (deg/sec)"].values.astype(np.float64))

    # 2. Gravity Correction & Leveling
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    g_norm = np.where(g_norm < 1e-3, 9.80665, g_norm)
    uz = np.column_stack([gx / g_norm, gy / g_norm, gz / g_norm])  # Unit vertical vector

    a_lin = np.column_stack([ax - gx, ay - gy, az - gz])
    a_vert_mag = np.sum(a_lin * uz, axis=1, keepdims=True)
    a_horiz = a_lin - a_vert_mag * uz  # Projected strictly into 2D horizontal plane

    # 3. Phone-to-Vehicle Forward and Lateral Alignment via Covariance
    c0 = np.cov(a_horiz[:, 0], v_acc)[0, 1] if len(a_horiz) > 1 else 1.0
    c1 = np.cov(a_horiz[:, 1], v_acc)[0, 1] if len(a_horiz) > 1 else 0.0
    c2 = np.cov(a_horiz[:, 2], v_acc)[0, 1] if len(a_horiz) > 1 else 0.0
    fwd = np.array([c0, c1, c2])
    fwd_norm = np.linalg.norm(fwd)
    fwd = fwd / fwd_norm if fwd_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

    a_longitudinal = np.sum(a_horiz * fwd, axis=1)

    uz_mean = uz.mean(axis=0)
    uz_mean = uz_mean / (np.linalg.norm(uz_mean) + 1e-8)
    lat_dir = np.cross(uz_mean, fwd)
    lat_dir_norm = np.linalg.norm(lat_dir)
    lat_dir = lat_dir / lat_dir_norm if lat_dir_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    a_lateral = np.sum(a_horiz * lat_dir, axis=1)

    # 4. Vertical Gyroscope Alignment
    # Determine whether col 15, 16, or 17 tracks vehicle yaw rate
    c15 = np.corrcoef(g15, v_yaw_rate)[0, 1] if len(g15) > 1 else 0.0
    c16 = np.corrcoef(g16, v_yaw_rate)[0, 1] if len(g16) > 1 else 0.0
    c17 = np.corrcoef(g17, v_yaw_rate)[0, 1] if len(g17) > 1 else 0.0

    best_c = max(abs(c15), abs(c16), abs(c17))
    if best_c == abs(c15):
        raw_yaw = g15
        s_sign = 1.0 if c15 >= 0 else -1.0
    elif best_c == abs(c16):
        raw_yaw = g16
        s_sign = 1.0 if c16 >= 0 else -1.0
    else:
        raw_yaw = g17
        s_sign = 1.0 if c17 >= 0 else -1.0

    w_yaw = s_sign * raw_yaw

    # 5. Angular Acceleration via Savitzky-Golay
    w_len = len(w_yaw)
    sg_win = 9 if w_len >= 9 else (w_len if w_len % 2 == 1 else w_len - 1)
    if sg_win >= 5:
        w_yaw_accel = savgol_filter(w_yaw, window_length=sg_win, polyorder=2, deriv=1, delta=DT)
    else:
        w_yaw_accel = np.gradient(w_yaw, DT)

    # 6. Geodesic velocity from WGS-84 coordinates
    lats = df_v["Latitude (degrees)"].values.astype(np.float64)
    lons = df_v["Longitude (degrees)"].values.astype(np.float64)
    headings = df_v["Heading (degrees)"].values.astype(np.float64)

    disp_10hz = compute_vectorized_geodesic_disp(lats, lons)
    v_true_10hz = disp_10hz / DT

    return {
        "name": name,
        "a_fwd": a_longitudinal,
        "w_yaw": w_yaw,
        "a_lat": a_lateral,
        "w_yaw_accel": w_yaw_accel,
        "v_true": v_true_10hz,
        "w_true": v_yaw_rate,
        "lats": lats,
        "lons": lons,
        "headings": headings,
    }


def create_physical_windows(
    journeys: list[dict],
    window_size: int = WINDOW_SIZE,
    is_train: bool = True,
    specialist_filter: str = "ALL",  # 'TURNING', 'LONGITUDINAL', 'ALL'
):
    """
    Constructs 10 Hz sliding windows in PURE PHYSICAL UNITS.

    Channels:
      [0] a_fwd:         m/s^2 (forward vehicle acceleration)
      [1] w_yaw:         rad/s (vehicle yaw rate around gravity)
      [2] a_lat:         m/s^2 (lateral vehicle acceleration)
      [3] v_prev:        m/s (autoregressive speed feedback)
      [4] w_yaw_accel:   rad/s^2 (angular acceleration)
      [5] centripetal:   m/s^2 (a_lat - v_prev * w_yaw)

    Targets:
      y_delta_v: m/s (true speed change over 0.1s, zero-centered)
      y_w_true:  rad/s (true yaw rate, zero-centered)
      y_zupt:    binary standstill indicator {0, 1}
      y_b_accel: m/s^2 forward accel residual, zero-centered
      y_b_gyro:  rad/s gyro residual, zero-centered
    """
    X_list = []
    y_dv_list = []
    y_w_list = []
    y_z_list = []
    y_ba_list = []
    y_bw_list = []

    for j in journeys:
        a_fwd = np.clip(j["a_fwd"], *CLIP["accel_fwd"])
        w_yaw = np.clip(j["w_yaw"], *CLIP["gyro_yaw"])
        a_lat = np.clip(j["a_lat"], *CLIP["accel_lat"])
        w_accel = np.clip(j["w_yaw_accel"], *CLIP["yaw_accel"])
        v_true = np.clip(j["v_true"], *CLIP["velocity"])
        w_true = np.clip(j["w_true"], *CLIP["gyro_yaw"])

        n = len(v_true)
        if n <= window_size + 10:
            continue

        if is_train:
            v_prev_series = np.clip(v_true + np.random.normal(0.0, 0.20, size=n), 0.0, CLIP["velocity"][1])
        else:
            v_prev_series = v_true.copy()

        centripetal_res = np.clip(a_lat - v_prev_series * w_yaw, *CLIP["centripetal_residual"])

        stride = 2 if is_train else 4

        for t in range(window_size, n, stride):
            cur_w = abs(w_true[t])
            cur_dv = abs(v_true[t] - v_true[t - 1])
            cur_v = v_true[t]

            if specialist_filter == "TURNING":
                # Model A focuses on turns, curves, corners, and active lateral forces
                is_turn = (cur_w > 0.03) or (abs(a_lat[t]) > 0.35) or (abs(w_yaw[t]) > 0.03)
                if not is_turn and np.random.rand() > 0.15:
                    continue

            elif specialist_filter == "LONGITUDINAL":
                # Model B focuses on throttle, braking, cruising, and stops
                is_long = (cur_dv > 0.04) or (cur_v > 10.0) or (cur_v < 0.8) or (abs(a_fwd[t]) > 0.4)
                if cur_w > 0.35 and abs(a_lat[t]) > 3.0:
                    continue
                if not is_long and np.random.rand() > 0.20:
                    continue

            ch_a_fwd = a_fwd[t - window_size + 1: t + 1]
            ch_w_yaw = w_yaw[t - window_size + 1: t + 1]
            ch_a_lat = a_lat[t - window_size + 1: t + 1]
            ch_v_prev = v_prev_series[t - window_size: t]
            ch_w_accel = w_accel[t - window_size + 1: t + 1]
            ch_centripetal = centripetal_res[t - window_size + 1: t + 1]

            win = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
            X_list.append(win)

            # Targets in clean physical units
            delta_v = np.clip(v_true[t] - v_true[t - 1], *CLIP["delta_v"])
            y_dv_list.append(delta_v)
            y_w_list.append(w_true[t])

            is_stopped = 1.0 if (v_true[t] < 0.25 and abs(a_fwd[t]) < 0.20) else 0.0
            y_z_list.append(is_stopped)

            accel_res = np.clip(a_fwd[t] - (v_true[t] - v_true[t - 1]) / DT, -1.0, 1.0)
            gyro_res = np.clip(w_yaw[t] - w_true[t], -0.2, 0.2)
            y_ba_list.append(accel_res)
            y_bw_list.append(gyro_res)

    X = np.asarray(X_list, dtype=np.float32)
    y_dv = np.asarray(y_dv_list, dtype=np.float32).reshape(-1, 1)
    y_w = np.asarray(y_w_list, dtype=np.float32).reshape(-1, 1)
    y_z = np.asarray(y_z_list, dtype=np.float32).reshape(-1, 1)
    y_ba = np.asarray(y_ba_list, dtype=np.float32).reshape(-1, 1)
    y_bw = np.asarray(y_bw_list, dtype=np.float32).reshape(-1, 1)

    return X, y_dv, y_w, y_z, y_ba, y_bw


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("PINO-DR: Step 1 Preprocessing Audit & Dataset Assembly")
    print("=" * 90)

    # 1. Parse descriptions
    pdf_path = Path("C:/Users/tiwar/OneDrive/Desktop/DATASETS/IO-VNBD/IO-VNBD/README_1.pdf")
    journey_desc = parse_journey_descriptions(pdf_path)

    # 2. Synchronized journey discovery
    pairs = find_synchronized_pairs()
    all_names = sorted(list(pairs.keys()))
    print(f"[Preprocess] Discovered {len(all_names)} synchronized journey pairs in IO-VNBD.")

    # 3. Strict Frozen Trip Split (50 Train / 9 Val / 13 Test)
    test_named_tags = set()
    for tags in TEST_SCENARIOS.values():
        for tag in tags:
            test_named_tags.add(tag.lower().replace("-", "").replace("_", ""))

    test_names = [k for k in all_names if k in test_named_tags]
    remaining = [k for k in all_names if k not in test_named_tags]

    np.random.seed(CONFIG["dataset"]["split_seed"])
    indices = np.random.permutation(len(remaining))
    pool = [remaining[i] for i in indices]

    idx = 0
    while len(test_names) < 13:
        test_names.append(pool[idx])
        idx += 1

    val_names = []
    while len(val_names) < 9:
        val_names.append(pool[idx])
        idx += 1

    train_names = pool[idx:]

    print(f"[Preprocess] Frozen Splits: {len(train_names)} Train / {len(val_names)} Val / {len(test_names)} Test trips")
    assert set(train_names).isdisjoint(set(val_names)), "Train/Val overlap!"
    assert set(train_names).isdisjoint(set(test_names)), "Train/Test overlap!"
    assert set(val_names).isdisjoint(set(test_names)), "Val/Test overlap!"

    # Process all journeys
    print("[Preprocess] Loading, synchronizing, and calibrating 10 Hz journey data...")
    train_journeys_all = []
    val_journeys_all = []
    test_journeys_all = []

    for name in all_names:
        s_p, v_p = pairs[name]
        j = process_journey_specialists(s_p, v_p, name=name)
        if j is None:
            continue
        if name in train_names:
            train_journeys_all.append(j)
        elif name in val_names:
            val_journeys_all.append(j)
        elif name in test_names:
            test_journeys_all.append(j)

    print(f"[Preprocess] Valid journeys loaded: {len(train_journeys_all)} Train, {len(val_journeys_all)} Val, {len(test_journeys_all)} Test")

    # 4. Classify journeys into Specialist Families
    def is_turning_trip(name):
        desc = journey_desc.get(name.lower(), "").lower()
        return ("round" in desc and "about" in desc) or ("sharp" in desc) or ("turns" in desc) or ("turn" in desc) or (name.lower() in ["vw6", "vw7", "vw8", "vta11", "vw5"])

    def is_longitudinal_trip(name):
        desc = journey_desc.get(name.lower(), "").lower()
        return ("motorway" in desc) or ("hard brake" in desc) or ("brake" in desc) or ("speed" in desc) or ("accel" in desc) or ("straight" in desc) or (name.lower() in ["vw12", "vta12", "vta9", "vw16b", "vw17", "vtb1", "vta15"])

    train_A_journeys = [j for j in train_journeys_all if is_turning_trip(j["name"])]
    train_B_journeys = [j for j in train_journeys_all if is_longitudinal_trip(j["name"])]
    val_A_journeys = [j for j in val_journeys_all if is_turning_trip(j["name"])]
    val_B_journeys = [j for j in val_journeys_all if is_longitudinal_trip(j["name"])]

    # If family list is too small, supplement with all journeys using the specialist window filter
    if len(train_A_journeys) < 8:
        train_A_journeys = train_journeys_all
    if len(train_B_journeys) < 8:
        train_B_journeys = train_journeys_all

    print(f"[Preprocess] Family A (Turning):      {len(train_A_journeys)} Train trips, {len(val_A_journeys)} Val trips")
    print(f"[Preprocess] Family B (Longitudinal): {len(train_B_journeys)} Train trips, {len(val_B_journeys)} Val trips")

    # 5. Extract Physical Windows
    print("\n[Preprocess] Creating physical temporal windows (W=20, dt=0.1s)...")
    X_tr_A, y_dv_tr_A, y_w_tr_A, y_z_tr_A, y_ba_tr_A, y_bw_tr_A = create_physical_windows(
        train_A_journeys, is_train=True, specialist_filter="TURNING"
    )
    X_va_A, y_dv_va_A, y_w_va_A, y_z_va_A, y_ba_va_A, y_bw_va_A = create_physical_windows(
        val_A_journeys, is_train=False, specialist_filter="TURNING"
    )

    X_tr_B, y_dv_tr_B, y_w_tr_B, y_z_tr_B, y_ba_tr_B, y_bw_tr_B = create_physical_windows(
        train_B_journeys, is_train=True, specialist_filter="LONGITUDINAL"
    )
    X_va_B, y_dv_va_B, y_w_va_B, y_z_va_B, y_ba_va_B, y_bw_va_B = create_physical_windows(
        val_B_journeys, is_train=False, specialist_filter="LONGITUDINAL"
    )

    print(f"  Model A (Turning)      -> Train windows: {X_tr_A.shape[0]:,}, Val windows: {X_va_A.shape[0]:,}")
    print(f"  Model B (Longitudinal) -> Train windows: {X_tr_B.shape[0]:,}, Val windows: {X_va_B.shape[0]:,}")

    # 6. Physical Augmentation on Training data ONLY (BEFORE scaling!)
    print("\n[Preprocess] Applying physical-unit IMU augmentation to training splits...")
    augmentor = IMUAugmentor()

    aug_X_tr_A = []
    for i in range(len(X_tr_A)):
        if np.random.rand() < 0.5:
            aug_X_tr_A.append(augmentor.augment_window(X_tr_A[i]))
        else:
            aug_X_tr_A.append(X_tr_A[i])
    X_tr_A_phys = np.asarray(aug_X_tr_A, dtype=np.float32)

    aug_X_tr_B = []
    for i in range(len(X_tr_B)):
        if np.random.rand() < 0.5:
            aug_X_tr_B.append(augmentor.augment_window(X_tr_B[i]))
        else:
            aug_X_tr_B.append(X_tr_B[i])
    X_tr_B_phys = np.asarray(aug_X_tr_B, dtype=np.float32)

    # 7. Channel-Wise Scalers fitted strictly on training data
    # Features X: shape (N * 20, 6) -> 6 feature channels across all time steps
    print("[Preprocess] Fitting channel-wise feature scalers (preserving temporal invariance)...")
    s_X_A = StandardScaler().fit(X_tr_A_phys.reshape(-1, 6))
    s_X_B = StandardScaler().fit(X_tr_B_phys.reshape(-1, 6))

    # Signed targets: zero-mean StandardScaler (with_mean=False) so 0.0 maps to 0.0
    print("[Preprocess] Fitting zero-centered target scalers (with_mean=False)...")
    s_ydv_A = StandardScaler(with_mean=False).fit(y_dv_tr_A)
    s_yw_A = StandardScaler(with_mean=False).fit(y_w_tr_A)
    s_yba_A = StandardScaler(with_mean=False).fit(y_ba_tr_A)
    s_ybw_A = StandardScaler(with_mean=False).fit(y_bw_tr_A)

    s_ydv_B = StandardScaler(with_mean=False).fit(y_dv_tr_B)
    s_yw_B = StandardScaler(with_mean=False).fit(y_w_tr_B)
    s_yba_B = StandardScaler(with_mean=False).fit(y_ba_tr_B)
    s_ybw_B = StandardScaler(with_mean=False).fit(y_bw_tr_B)

    # Transform Model A
    X_tr_A_s = s_X_A.transform(X_tr_A_phys.reshape(-1, 6)).reshape(X_tr_A_phys.shape).astype(np.float32)
    X_va_A_s = s_X_A.transform(X_va_A.reshape(-1, 6)).reshape(X_va_A.shape).astype(np.float32)
    y_dv_tr_A_s = s_ydv_A.transform(y_dv_tr_A).astype(np.float32)
    y_dv_va_A_s = s_ydv_A.transform(y_dv_va_A).astype(np.float32)
    y_w_tr_A_s = s_yw_A.transform(y_w_tr_A).astype(np.float32)
    y_w_va_A_s = s_yw_A.transform(y_w_va_A).astype(np.float32)
    y_ba_tr_A_s = s_yba_A.transform(y_ba_tr_A).astype(np.float32)
    y_ba_va_A_s = s_yba_A.transform(y_ba_va_A).astype(np.float32)
    y_bw_tr_A_s = s_ybw_A.transform(y_bw_tr_A).astype(np.float32)
    y_bw_va_A_s = s_ybw_A.transform(y_bw_va_A).astype(np.float32)

    # Transform Model B
    X_tr_B_s = s_X_B.transform(X_tr_B_phys.reshape(-1, 6)).reshape(X_tr_B_phys.shape).astype(np.float32)
    X_va_B_s = s_X_B.transform(X_va_B.reshape(-1, 6)).reshape(X_va_B.shape).astype(np.float32)
    y_dv_tr_B_s = s_ydv_B.transform(y_dv_tr_B).astype(np.float32)
    y_dv_va_B_s = s_ydv_B.transform(y_dv_va_B).astype(np.float32)
    y_w_tr_B_s = s_yw_B.transform(y_w_tr_B).astype(np.float32)
    y_w_va_B_s = s_yw_B.transform(y_w_va_B).astype(np.float32)
    y_ba_tr_B_s = s_yba_B.transform(y_ba_tr_B).astype(np.float32)
    y_ba_va_B_s = s_yba_B.transform(y_ba_va_B).astype(np.float32)
    y_bw_tr_B_s = s_ybw_B.transform(y_bw_tr_B).astype(np.float32)
    y_bw_va_B_s = s_ybw_B.transform(y_bw_va_B).astype(np.float32)

    # 8. Save compressed specialist datasets
    npz_path = DATA_DIR / "dataset_specialists.npz"
    np.savez_compressed(
        npz_path,
        # Model A
        X_tr_A=X_tr_A_s, y_dv_tr_A=y_dv_tr_A_s, y_w_tr_A=y_w_tr_A_s, y_z_tr_A=y_z_tr_A, y_ba_tr_A=y_ba_tr_A_s, y_bw_tr_A=y_bw_tr_A_s,
        X_va_A=X_va_A_s, y_dv_va_A=y_dv_va_A_s, y_w_va_A=y_w_va_A_s, y_z_va_A=y_z_va_A, y_ba_va_A=y_ba_va_A_s, y_bw_va_A=y_bw_va_A_s,
        # Model B
        X_tr_B=X_tr_B_s, y_dv_tr_B=y_dv_tr_B_s, y_w_tr_B=y_w_tr_B_s, y_z_tr_B=y_z_tr_B, y_ba_tr_B=y_ba_tr_B_s, y_bw_tr_B=y_bw_tr_B_s,
        X_va_B=X_va_B_s, y_dv_va_B=y_dv_va_B_s, y_w_va_B=y_w_va_B_s, y_z_va_B=y_z_va_B, y_ba_va_B=y_ba_va_B_s, y_bw_va_B=y_bw_va_B_s,
    )
    print(f"[Preprocess] Saved specialist dataset splits to {npz_path}")

    # Save scalers
    scalers_specialists = {
        "A": {
            "X": s_X_A,
            "y_dv": s_ydv_A,
            "y_w": s_yw_A,
            "y_ba": s_yba_A,
            "y_bw": s_ybw_A,
        },
        "B": {
            "X": s_X_B,
            "y_dv": s_ydv_B,
            "y_w": s_yw_B,
            "y_ba": s_yba_B,
            "y_bw": s_ybw_B,
        },
    }
    with open(DATA_DIR / "scalers_specialists.pkl", "wb") as f:
        pickle.dump(scalers_specialists, f)
    print(f"[Preprocess] Saved scalers to {DATA_DIR / 'scalers_specialists.pkl'}")

    # 9. Save validation scenarios for closed-loop validation drift evaluation
    val_scenarios = {
        "turning": val_A_journeys,
        "longitudinal": val_B_journeys,
        "all": val_journeys_all,
    }
    with open(DATA_DIR / "val_scenarios_specialists.pkl", "wb") as f:
        pickle.dump(val_scenarios, f)
    print(f"[Preprocess] Saved validation scenario journeys to {DATA_DIR / 'val_scenarios_specialists.pkl'}")

    # 10. Save test scenarios (FROZEN - strictly for Step 9 final reporting)
    # Group test journeys by scenario
    test_scenarios_grouped = {
        "motorway": [j for j in test_journeys_all if "vw12" in j["name"].lower()],
        "roundabout": [j for j in test_journeys_all if "vta11" in j["name"].lower()],
        "quick_accel": [j for j in test_journeys_all if "vta12" in j["name"].lower()],
        "hard_brake": [j for j in test_journeys_all if any(tag in j["name"].lower() for tag in ["vw16b", "vw17", "vta9"])],
        "sharp_turns": [j for j in test_journeys_all if any(tag in j["name"].lower() for tag in ["vw6", "vw7", "vw8"])],
        "all": test_journeys_all,
    }
    with open(DATA_DIR / "test_scenarios_v6.pkl", "wb") as f:
        pickle.dump(test_scenarios_grouped, f)
    print(f"[Preprocess] Saved frozen test scenarios to {DATA_DIR / 'test_scenarios_v6.pkl'}")

    print("=" * 90)
    print("[Preprocess] Step 1 Audit & Preprocessing Complete. Zero Leakage Guaranteed.")
    print("=" * 90)


if __name__ == "__main__":
    main()
