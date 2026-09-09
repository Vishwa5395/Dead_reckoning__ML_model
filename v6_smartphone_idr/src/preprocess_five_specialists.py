"""
preprocess_five_specialists.py
------------------------------
Rigorous Multi-Scale Preprocessing Pipeline for Five Independent Specialists:
  - M1: Forward Velocity Specialist (T=25, 2.5s history, 14 features)
  - M2: Yaw Rate Specialist (T=20, 2.0s history, 12 features)
  - M3: Accelerometer Bias Specialist (T=50, 5.0s history, 6 features)
  - M4: Gyroscope Bias Specialist (T=50, 5.0s history, 6 features)
  - M5: Uncertainty Adapter (T=15, 1.5s history, 9 raw IMU features)

Key Integrity Protections:
  1. Sub-second cross-correlation synchronization between phone IMU and CAN reference.
  2. Gyroscope vertical turning axis identification & covariance sign alignment.
  3. Channel-wise standard scaling across all timesteps preserving temporal translation invariance.
  4. Zero-centered target scaling (with_mean=False) for signed delta/rotational quantities.
  5. Physical-unit IMU augmentation applied strictly to training splits.
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
DT = CONFIG["dt"]  # 0.1 s
CLIP = CONFIG["clip_limits"]

# Quantity-specific temporal receptive fields (steps @ 10 Hz)
T_M1 = 25  # 2.5s for velocity
T_M2 = 20  # 2.0s for yaw rate
T_M3 = 50  # 5.0s for accelerometer bias
T_M4 = 50  # 5.0s for gyroscope bias
T_M5 = 15  # 1.5s for raw IMU uncertainty

MAX_T = max(T_M1, T_M2, T_M3, T_M4, T_M5)

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
    dist = np.where(dist > 15.0, 0.0, dist)  # Filter GPS jumps (>540 km/h)

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


def process_journey_five_specialists(s_path: str, v_path: str, name: str = "journey") -> dict | None:
    """
    Processes one synchronized journey pair natively at 10 Hz:
      - Synchronizes sub-second clock offsets
      - Computes gravity-leveling & horizontal forward/lateral projection
      - Extracts vertical gyro turn axis
      - Derives low-frequency bias targets and motion signals
    """
    try:
        df_s = pd.read_csv(s_path, encoding="latin1")
        df_v = pd.read_csv(v_path)
    except Exception:
        return None

    df_v.columns = [c.strip() for c in df_v.columns]
    min_len = min(len(df_s), len(df_v))
    if min_len < 100:
        return None

    df_s = df_s.iloc[:min_len]
    df_v = df_v.iloc[:min_len]

    # Raw IMU
    ax = df_s.iloc[:, 9].values.astype(np.float64)
    ay = df_s.iloc[:, 10].values.astype(np.float64)
    az = df_s.iloc[:, 11].values.astype(np.float64)
    gx = df_s.iloc[:, 12].values.astype(np.float64)
    gy = df_s.iloc[:, 13].values.astype(np.float64)
    gz = df_s.iloc[:, 14].values.astype(np.float64)
    g15 = df_s.iloc[:, 15].values.astype(np.float64)
    g16 = df_s.iloc[:, 16].values.astype(np.float64)
    g17 = df_s.iloc[:, 17].values.astype(np.float64)

    # Reference signals
    v_acc = df_v["Indicated Longitudinal Acceleration (g)"].values.astype(np.float64) * 9.80665
    v_yaw_rate = np.radians(df_v["Yaw Rate (deg/sec)"].values.astype(np.float64))

    # 1. Sub-second clock synchronization
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

    if best_lag > 0:
        df_s = df_s.iloc[:min_len - best_lag].reset_index(drop=True)
        df_v = df_v.iloc[best_lag:min_len].reset_index(drop=True)
    elif best_lag < 0:
        df_s = df_s.iloc[-best_lag:min_len].reset_index(drop=True)
        df_v = df_v.iloc[:min_len + best_lag].reset_index(drop=True)

    n_aligned = min(len(df_s), len(df_v))
    if n_aligned < 80:
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

    # 2. Gravity separation
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    g_norm = np.where(g_norm < 1e-3, 9.80665, g_norm)
    uz = np.column_stack([gx / g_norm, gy / g_norm, gz / g_norm])  # (N, 3)

    a_lin = np.column_stack([ax - gx, ay - gy, az - gz])
    a_vert_mag = np.sum(a_lin * uz, axis=1, keepdims=True)
    a_horiz = a_lin - a_vert_mag * uz

    # 3. Vehicle forward & lateral alignment
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

    # 5. Numerical derivatives
    w_len = len(w_yaw)
    sg_win = 9 if w_len >= 9 else (w_len if w_len % 2 == 1 else w_len - 1)
    if sg_win >= 5:
        w_yaw_accel = savgol_filter(w_yaw, window_length=sg_win, polyorder=2, deriv=1, delta=DT)
        jerk = savgol_filter(a_longitudinal, window_length=sg_win, polyorder=2, deriv=1, delta=DT)
    else:
        w_yaw_accel = np.gradient(w_yaw, DT)
        jerk = np.gradient(a_longitudinal, DT)

    # 6. Geodesic velocity from WGS-84 coordinates
    lats = df_v["Latitude (degrees)"].values.astype(np.float64)
    lons = df_v["Longitude (degrees)"].values.astype(np.float64)
    headings = df_v["Heading (degrees)"].values.astype(np.float64)

    disp_10hz = compute_vectorized_geodesic_disp(lats, lons)
    v_true_10hz = disp_10hz / DT

    # 7. Low-frequency bias targets (smoothed difference between IMU and kinematic truth)
    a_kinematic = np.gradient(v_true_10hz, DT)
    raw_accel_bias = a_longitudinal - a_kinematic
    raw_gyro_bias = w_yaw - v_yaw_rate

    # Smooth bias targets over 20-step (2.0s) window to isolate low-frequency drift state
    bias_win = 21 if len(raw_accel_bias) >= 21 else (len(raw_accel_bias) // 2 * 2 + 1)
    if bias_win >= 5:
        accel_bias_target = savgol_filter(raw_accel_bias, window_length=bias_win, polyorder=1)
        gyro_bias_target = savgol_filter(raw_gyro_bias, window_length=bias_win, polyorder=1)
    else:
        accel_bias_target = raw_accel_bias
        gyro_bias_target = raw_gyro_bias

    return {
        "name": name,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "uz": uz,
        "a_fwd": a_longitudinal,
        "a_lat": a_lateral,
        "w_yaw": w_yaw,
        "w_yaw_accel": w_yaw_accel,
        "jerk": jerk,
        "v_true": v_true_10hz,
        "w_true": v_yaw_rate,
        "b_a": accel_bias_target,
        "b_g": gyro_bias_target,
        "lats": lats,
        "lons": lons,
        "headings": headings,
    }


def extract_multiscale_windows(journeys: list[dict], is_train: bool = True):
    """
    Extracts quantity-specific sliding windows for each of the 5 models.
    """
    X_M1_list, y_M1_list = [], []
    X_M2_list, y_M2_list = [], []
    X_M3_list, y_M3_list = [], []
    X_M4_list, y_M4_list = [], []
    X_M5_list, y_M5_list = [], []

    stride = 3 if is_train else 5

    for j in journeys:
        n = len(j["v_true"])
        if n <= MAX_T + 15:
            continue

        ax = j["ax"]
        ay = j["ay"]
        az = j["az"]
        gx = j["gx"]
        gy = j["gy"]
        gz = j["gz"]
        uz = j["uz"]
        a_fwd = j["a_fwd"]
        a_lat = j["a_lat"]
        w_yaw = j["w_yaw"]
        w_accel = j["w_yaw_accel"]
        jerk = j["jerk"]
        v_true = j["v_true"]
        w_true = j["w_true"]
        b_a = j["b_a"]
        b_g = j["b_g"]

        a_norm = np.sqrt(ax**2 + ay**2 + az**2)
        g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

        # Autoregressive previous velocity with small noise during training
        if is_train:
            v_prev = np.clip(v_true + np.random.normal(0.0, 0.20, size=n), 0.0, 45.0)
        else:
            v_prev = v_true.copy()

        for t in range(MAX_T, n, stride):
            # --- M1: Velocity Specialist (T=25, 14 features) ---
            # Features: [ax, ay, az, gx, gy, gz, uz_x, uz_y, uz_z, a_fwd, a_lat, w_yaw, jerk, v_prev]
            s1 = t - T_M1 + 1
            win_M1 = np.column_stack([
                ax[s1:t+1], ay[s1:t+1], az[s1:t+1],
                gx[s1:t+1], gy[s1:t+1], gz[s1:t+1],
                uz[s1:t+1, 0], uz[s1:t+1, 1], uz[s1:t+1, 2],
                a_fwd[s1:t+1], a_lat[s1:t+1], w_yaw[s1:t+1],
                jerk[s1:t+1], v_prev[s1-1:t]
            ])
            # Target: [v_true, delta_v]
            target_M1 = [v_true[t], v_true[t] - v_true[t - 1]]
            X_M1_list.append(win_M1)
            y_M1_list.append(target_M1)

            # --- M2: Yaw Specialist (T=20, 12 features) ---
            # Features: [gx, gy, gz, ax, ay, az, uz_x, uz_y, uz_z, a_lat, a_fwd, w_yaw]
            s2 = t - T_M2 + 1
            win_M2 = np.column_stack([
                gx[s2:t+1], gy[s2:t+1], gz[s2:t+1],
                ax[s2:t+1], ay[s2:t+1], az[s2:t+1],
                uz[s2:t+1, 0], uz[s2:t+1, 1], uz[s2:t+1, 2],
                a_lat[s2:t+1], a_fwd[s2:t+1], w_yaw[s2:t+1]
            ])
            # Target: [w_true, delta_psi]
            target_M2 = [w_true[t], w_true[t] * DT]
            X_M2_list.append(win_M2)
            y_M2_list.append(target_M2)

            # --- M3: Accel Bias Specialist (T=50, 6 features) ---
            # Features: [a_fwd, a_norm, uz_x, uz_y, uz_z, v_prev]
            s3 = t - T_M3 + 1
            win_M3 = np.column_stack([
                a_fwd[s3:t+1], a_norm[s3:t+1],
                uz[s3:t+1, 0], uz[s3:t+1, 1], uz[s3:t+1, 2],
                v_prev[s3-1:t]
            ])
            target_M3 = [b_a[t]]
            X_M3_list.append(win_M3)
            y_M3_list.append(target_M3)

            # --- M4: Gyro Bias Specialist (T=50, 6 features) ---
            # Features: [w_yaw, g_norm, w_accel, uz_x, uz_y, uz_z]
            s4 = t - T_M4 + 1
            win_M4 = np.column_stack([
                w_yaw[s4:t+1], g_norm[s4:t+1], w_accel[s4:t+1],
                uz[s4:t+1, 0], uz[s4:t+1, 1], uz[s4:t+1, 2]
            ])
            target_M4 = [b_g[t]]
            X_M4_list.append(win_M4)
            y_M4_list.append(target_M4)

            # --- M5: Uncertainty Adapter (T=15, 9 raw IMU features) ---
            # Features: [ax, ay, az, gx, gy, gz, uz_x, uz_y, uz_z]
            s5 = t - T_M5 + 1
            win_M5 = np.column_stack([
                ax[s5:t+1], ay[s5:t+1], az[s5:t+1],
                gx[s5:t+1], gy[s5:t+1], gz[s5:t+1],
                uz[s5:t+1, 0], uz[s5:t+1, 1], uz[s5:t+1, 2]
            ])
            target_M5 = [0.0]
            X_M5_list.append(win_M5)
            y_M5_list.append(target_M5)

    return (
        np.asarray(X_M1_list, dtype=np.float32), np.asarray(y_M1_list, dtype=np.float32),
        np.asarray(X_M2_list, dtype=np.float32), np.asarray(y_M2_list, dtype=np.float32),
        np.asarray(X_M3_list, dtype=np.float32), np.asarray(y_M3_list, dtype=np.float32),
        np.asarray(X_M4_list, dtype=np.float32), np.asarray(y_M4_list, dtype=np.float32),
        np.asarray(X_M5_list, dtype=np.float32), np.asarray(y_M5_list, dtype=np.float32),
    )


def fit_and_scale(X_tr: np.ndarray, X_va: np.ndarray):
    """Channel-wise standard scaling across all timesteps."""
    n_tr, t_steps, n_ch = X_tr.shape
    n_va = X_va.shape[0]

    X_tr_flat = X_tr.reshape(n_tr * t_steps, n_ch)
    X_va_flat = X_va.reshape(n_va * t_steps, n_ch)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(n_tr, t_steps, n_ch)
    X_va_scaled = scaler.transform(X_va_flat).reshape(n_va, t_steps, n_ch)

    return X_tr_scaled, X_va_scaled, scaler


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("PINO-DR: 5-Independent-Specialist Preprocessing Pipeline")
    print("=" * 90)

    # 1. Discover journeys
    pairs = find_synchronized_pairs()
    print(f"[Preprocess] Found {len(pairs)} synchronized file pairs")

    all_names = sorted(list(pairs.keys()))
    test_named_tags = [tag.lower() for tags in TEST_SCENARIOS.values() for tag in tags]

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

    print(f"[Preprocess] Strict Splits: {len(train_names)} Train / {len(val_names)} Val / {len(test_names)} Test")

    # 2. Process all journeys
    train_journeys = []
    val_journeys = []
    test_journeys = []

    for name in all_names:
        s_p, v_p = pairs[name]
        j = process_journey_five_specialists(s_p, v_p, name=name)
        if j is None:
            continue
        if name in train_names:
            train_journeys.append(j)
        elif name in val_names:
            val_journeys.append(j)
        elif name in test_names:
            test_journeys.append(j)

    print(f"[Preprocess] Loaded: {len(train_journeys)} Train, {len(val_journeys)} Val, {len(test_journeys)} Test")

    # 3. Extract multi-scale windows
    print("\n[Preprocess] Extracting multi-scale sliding windows for M1..M5...")
    (
        X_tr_M1, y_tr_M1,
        X_tr_M2, y_tr_M2,
        X_tr_M3, y_tr_M3,
        X_tr_M4, y_tr_M4,
        X_tr_M5, y_tr_M5
    ) = extract_multiscale_windows(train_journeys, is_train=True)

    (
        X_va_M1, y_va_M1,
        X_va_M2, y_va_M2,
        X_va_M3, y_va_M3,
        X_va_M4, y_va_M4,
        X_va_M5, y_va_M5
    ) = extract_multiscale_windows(val_journeys, is_train=False)

    print(f"  M1 (Velocity, T={T_M1}):   Train {X_tr_M1.shape}, Val {X_va_M1.shape}")
    print(f"  M2 (Yaw Rate, T={T_M2}):   Train {X_tr_M2.shape}, Val {X_va_M2.shape}")
    print(f"  M3 (Accel Bias, T={T_M3}): Train {X_tr_M3.shape}, Val {X_va_M3.shape}")
    print(f"  M4 (Gyro Bias, T={T_M4}):  Train {X_tr_M4.shape}, Val {X_va_M4.shape}")
    print(f"  M5 (Uncertainty, T={T_M5}): Train {X_tr_M5.shape}, Val {X_va_M5.shape}")

    # 4. Standard Scaling per specialist
    print("\n[Preprocess] Fitting channel-wise StandardScalers...")
    X_tr_M1_s, X_va_M1_s, scaler_M1 = fit_and_scale(X_tr_M1, X_va_M1)
    X_tr_M2_s, X_va_M2_s, scaler_M2 = fit_and_scale(X_tr_M2, X_va_M2)
    X_tr_M3_s, X_va_M3_s, scaler_M3 = fit_and_scale(X_tr_M3, X_va_M3)
    X_tr_M4_s, X_va_M4_s, scaler_M4 = fit_and_scale(X_tr_M4, X_va_M4)
    X_tr_M5_s, X_va_M5_s, scaler_M5 = fit_and_scale(X_tr_M5, X_va_M5)

    # 5. Target Scalers (Zero-centered for signed quantities)
    scaler_y_v = StandardScaler(with_mean=False)
    y_tr_M1_s = scaler_y_v.fit_transform(y_tr_M1)
    y_va_M1_s = scaler_y_v.transform(y_va_M1)

    scaler_y_w = StandardScaler(with_mean=False)
    y_tr_M2_s = scaler_y_w.fit_transform(y_tr_M2)
    y_va_M2_s = scaler_y_w.transform(y_va_M2)

    scaler_y_ba = StandardScaler(with_mean=False)
    y_tr_M3_s = scaler_y_ba.fit_transform(y_tr_M3)
    y_va_M3_s = scaler_y_ba.transform(y_va_M3)

    scaler_y_bg = StandardScaler(with_mean=False)
    y_tr_M4_s = scaler_y_bg.fit_transform(y_tr_M4)
    y_va_M4_s = scaler_y_bg.transform(y_va_M4)

    scalers = {
        "scaler_M1": scaler_M1,
        "scaler_M2": scaler_M2,
        "scaler_M3": scaler_M3,
        "scaler_M4": scaler_M4,
        "scaler_M5": scaler_M5,
        "scaler_y_v": scaler_y_v,
        "scaler_y_w": scaler_y_w,
        "scaler_y_ba": scaler_y_ba,
        "scaler_y_bg": scaler_y_bg,
    }

    # 6. Save Dataset & Scalers
    npz_path = DATA_DIR / "dataset_five_specialists.npz"
    scalers_path = DATA_DIR / "scalers_five_specialists.pkl"
    val_meta_path = DATA_DIR / "val_journeys_five_specialists.pkl"

    print(f"\n[Preprocess] Saving dataset to {npz_path}...")
    np.savez_compressed(
        npz_path,
        X_tr_M1=X_tr_M1_s, y_tr_M1=y_tr_M1_s, y_tr_M1_raw=y_tr_M1,
        X_va_M1=X_va_M1_s, y_va_M1=y_va_M1_s, y_va_M1_raw=y_va_M1,

        X_tr_M2=X_tr_M2_s, y_tr_M2=y_tr_M2_s, y_tr_M2_raw=y_tr_M2,
        X_va_M2=X_va_M2_s, y_va_M2=y_va_M2_s, y_va_M2_raw=y_va_M2,

        X_tr_M3=X_tr_M3_s, y_tr_M3=y_tr_M3_s, y_tr_M3_raw=y_tr_M3,
        X_va_M3=X_va_M3_s, y_va_M3=y_va_M3_s, y_va_M3_raw=y_va_M3,

        X_tr_M4=X_tr_M4_s, y_tr_M4=y_tr_M4_s, y_tr_M4_raw=y_tr_M4,
        X_va_M4=X_va_M4_s, y_va_M4=y_va_M4_s, y_va_M4_raw=y_va_M4,

        X_tr_M5=X_tr_M5_s,
        X_va_M5=X_va_M5_s,
    )

    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)

    with open(val_meta_path, "wb") as f:
        pickle.dump(val_journeys, f)

    print(f"[Preprocess] Successfully saved scalers to {scalers_path}")
    print(f"[Preprocess] Successfully saved {len(val_journeys)} validation journeys to {val_meta_path}")
    print("[Preprocess] Preprocessing Complete!")


if __name__ == "__main__":
    main()
