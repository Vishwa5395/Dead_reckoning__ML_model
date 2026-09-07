"""
preprocess_v5.py
----------------
Data preparation for PINO-DR v5 (Wheel-Aided Dead Reckoning).

Step 1:
- Extends V-*.csv parsing to extract Wheel Speed Rear Left & Right (rad/sec).
- Computes rear-axle linear wheel speed v_wheel = r_tire * 0.5 * (w_rl + w_rr).
- Strictly uses train-only calibrated tire radius: r_tire = 0.277209 m.
- 7-channel temporal tensor:
  [a_fwd, w_yaw, a_lat_measured, v_prev, w_yaw_accel, centripetal_residual, v_wheel]
- Exact frozen V3/V4 trip-level split (50 train / 9 val / 13 test journeys, zero temporal leakage).
- Scalers fitted strictly on train split.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# ─── Project paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASET_BASE_DIR = r"C:\Users\tiwar\OneDrive\Desktop\DATASETS\IO-VNBD\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

# Named test scenario journey tags (identical to v3/v4 / Onyekpe et al.)
TEST_SCENARIOS = {
    'motorway': ['Vw12'],
    'roundabout': ['Vta11'],
    'quick_accel': ['Vta12'],
    'hard_brake': ['Vw16b', 'Vw17', 'Vta9'],
    'sharp_turns': ['Vw6', 'Vw7', 'Vw8'],
}

# Physical clip thresholds (road vehicle limits)
CLIP = {
    "accel": (-8.0, 8.0),             # m/s² (longitudinal and lateral)
    "disp": (0.0, 45.0),              # m per 1-second step (equiv 0–162 km/h)
    "gyro": (-1.0, 1.0),              # rad/s
    "yaw": (-1.2, 1.2),               # rad/s
    "yaw_accel": (-2.0, 2.0),         # rad/s²
    "centripetal_res": (-8.0, 8.0),   # m/s²
    "v_wheel": (0.0, 45.0),           # m/s (equiv 0–162 km/h)
}

# Strictly train-only calibrated constants (Zero leakage)
R_TIRE_TRAIN = 0.277209  # meters, derived from 550,756 samples across 50 train trips
TRACK_WIDTH_TRAIN = 1.4785  # meters, rear track width fitted on train trips

WINDOW_SIZE = 10
NOISE_STD = 0.5   # Gaussian noise σ for v_prev during training


# ─── Vincenty geodesic distance ──────────────────────────────────────────────

def vincenty_dist(p1, p2):
    """Geodesic distance in meters between two (lat, lon) tuples."""
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    a, b, f = 6378137.0, 6356752.314245, 1 / 298.257223563
    L = lon2 - lon1
    U1 = math.atan((1 - f) * math.tan(lat1))
    U2 = math.atan((1 - f) * math.tan(lat2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    lambda_ = L
    for _ in range(200):
        sinLambda, cosLambda = math.sin(lambda_), math.cos(lambda_)
        sinSigma = math.sqrt(
            (cosU2 * sinLambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2
        )
        if sinSigma == 0:
            return 0.0
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = (cosU1 * cosU2 * sinLambda) / sinSigma
        cosSqAlpha = 1.0 - sinAlpha ** 2
        cos2SigmaM = (cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha) if cosSqAlpha != 0 else 0.0
        C = (f / 16.0) * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        lambda_prev = lambda_
        lambda_ = L + (1 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2))
        )
        if abs(lambda_ - lambda_prev) < 1e-12:
            break
    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A_ = 1 + (uSq / 16384.0) * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B_ = (uSq / 1024.0) * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B_ * sinSigma * (
        cos2SigmaM + (B_ / 4.0) * (
            cosSigma * (-1 + 2 * cos2SigmaM ** 2)
            - (B_ / 6.0) * cos2SigmaM * (-3 + 4 * sinAlpha ** 2) * (-3 + 4 * cos2SigmaM ** 2)
        )
    )
    return b * A_ * (sigma - deltaSigma)


# ─── Sensor processing ───────────────────────────────────────────────────────

def find_synchronized_pairs(base_dir=DATASET_BASE_DIR):
    """Finds all synchronized (S-*.csv, V-*.csv) file pairs."""
    s_files, v_files = {}, {}
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.csv'):
                continue
            full = os.path.join(root, f)
            fl = f.lower()
            if fl.startswith('s-'):
                key = fl.replace('s-', '').replace('.csv', '').replace('-', '').replace('_', '')
                s_files[key] = full
            elif fl.startswith('v-'):
                key = fl.replace('v-', '').replace('.csv', '').replace('-', '').replace('_', '')
                v_files[key] = full
    paired = {}
    for key, s_p in s_files.items():
        if key in v_files:
            paired[key] = (s_p, v_files[key])
    return paired


def process_journey_v5(s_path, v_path, name="journey"):
    """
    Processes one synchronized journey into 1-second intervals.
    Returns enriched arrays including lateral acceleration, 10Hz smoothed
    yaw acceleration, and rear-axle wheel speed v_wheel.
    """
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path)
    df_v.columns = [c.strip() for c in df_v.columns]

    min_len = min(len(df_s), len(df_v))
    if min_len < 30:
        return None

    df_s = df_s.iloc[:min_len]
    df_v = df_v.iloc[:min_len]

    # ── Smartphone IMU extraction (S- file) ──
    ax = df_s.iloc[:, 9].values.astype(np.float64)
    ay = df_s.iloc[:, 10].values.astype(np.float64)
    az = df_s.iloc[:, 11].values.astype(np.float64)
    gx = df_s.iloc[:, 12].values.astype(np.float64)
    gy = df_s.iloc[:, 13].values.astype(np.float64)
    gz = df_s.iloc[:, 14].values.astype(np.float64)
    wx = df_s.iloc[:, 15].values.astype(np.float64)
    wy = df_s.iloc[:, 16].values.astype(np.float64)
    wz = df_s.iloc[:, 17].values.astype(np.float64)

    # Gravity leveling
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    g_norm = np.where(g_norm < 1e-3, 9.80665, g_norm)
    uz = np.column_stack([gx / g_norm, gy / g_norm, gz / g_norm])  # unit vertical

    a_lin = np.column_stack([ax - gx, ay - gy, az - gz])
    a_vert_mag = np.sum(a_lin * uz, axis=1, keepdims=True)
    a_horiz = a_lin - a_vert_mag * uz

    w_gyro = np.column_stack([wx, wy, wz])
    w_vert = np.sum(w_gyro * uz, axis=1)  # yaw rate around vertical

    # ── Forward axis estimation via covariance with V-file ECU accel ──
    v_acc = df_v['Indicated Longitudinal Acceleration (g)'].values.astype(np.float64) * 9.80665
    v_yaw_rate = np.radians(df_v['Yaw Rate (deg/sec)'].values.astype(np.float64))

    c0 = np.cov(a_horiz[:, 0], v_acc)[0, 1] if len(a_horiz) > 1 else 1.0
    c1 = np.cov(a_horiz[:, 1], v_acc)[0, 1] if len(a_horiz) > 1 else 0.0
    c2 = np.cov(a_horiz[:, 2], v_acc)[0, 1] if len(a_horiz) > 1 else 0.0
    fwd = np.array([c0, c1, c2])
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm > 1e-6:
        fwd = fwd / fwd_norm
    else:
        fwd = np.array([0.0, 1.0, 0.0])

    a_longitudinal = np.sum(a_horiz * fwd, axis=1)

    # ── Lateral acceleration ──
    uz_mean = uz.mean(axis=0)
    uz_mean = uz_mean / (np.linalg.norm(uz_mean) + 1e-8)
    lat_dir = np.cross(uz_mean, fwd)
    lat_dir_norm = np.linalg.norm(lat_dir)
    if lat_dir_norm > 1e-6:
        lat_dir = lat_dir / lat_dir_norm
    else:
        lat_dir = np.array([1.0, 0.0, 0.0])
    a_lateral = np.sum(a_horiz * lat_dir, axis=1)

    # ── Yaw sign alignment ──
    corr = np.corrcoef(w_vert, v_yaw_rate)[0, 1] if len(w_vert) > 1 else 1.0
    yaw_sign = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
    w_yaw_aligned = yaw_sign * w_vert

    # ── 10 Hz Smoothed Yaw Acceleration (Savitzky-Golay) ──
    dt_10hz = 0.1
    w_dot_raw = np.gradient(w_yaw_aligned, dt_10hz)
    if len(w_dot_raw) >= 7:
        w_yaw_accel_10hz = savgol_filter(w_dot_raw, window_length=5, polyorder=2)
    else:
        w_yaw_accel_10hz = w_dot_raw

    # ── Wheel speed extraction (rear axle average) ──
    w_rl_10hz = df_v['Wheel Speed Rear Left (rad/sec)'].values.astype(np.float64)
    w_rr_10hz = df_v['Wheel Speed Rear Right (rad/sec)'].values.astype(np.float64)
    w_rear_avg_10hz = 0.5 * (w_rl_10hz + w_rr_10hz)
    v_wheel_10hz = R_TIRE_TRAIN * w_rear_avg_10hz  # linear speed in m/s

    # Wheel differential yaw rate for reference cross-check
    w_diff_yaw_10hz = (R_TIRE_TRAIN / TRACK_WIDTH_TRAIN) * (w_rr_10hz - w_rl_10hz)

    # ── Downsample 10 Hz → 1 Hz ──
    n_sec = min_len // 10
    if n_sec < 12:
        return None

    a_fwd_1s = a_longitudinal[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_yaw_1s = w_yaw_aligned[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    a_lat_1s = a_lateral[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_yaw_accel_1s = w_yaw_accel_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    v_wheel_1s = v_wheel_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_diff_yaw_1s = w_diff_yaw_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_gps_1s = v_yaw_rate[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)

    lats_10hz = df_v['Latitude (degrees)'].values.astype(np.float64)
    lons_10hz = df_v['Longitude (degrees)'].values.astype(np.float64)
    heading_10hz = df_v['Heading (degrees)'].values.astype(np.float64)

    lats_1s = lats_10hz[np.arange(n_sec) * 10]
    lons_1s = lons_10hz[np.arange(n_sec) * 10]
    headings_1s = heading_10hz[np.arange(n_sec) * 10]

    # Geodesic displacement (m per 1-second step)
    x_gps_1s = np.zeros(n_sec, dtype=np.float64)
    for t in range(1, n_sec):
        p1 = (lats_1s[t - 1], lons_1s[t - 1])
        p2 = (lats_1s[t], lons_1s[t])
        dist = vincenty_dist(p1, p2)
        if dist is None or np.isnan(dist):
            dlat = (p2[0] - p1[0]) * 111139.0
            dlon = (p2[1] - p1[1]) * 111139.0 * np.cos(np.radians(p1[0]))
            dist = np.sqrt(dlat ** 2 + dlon ** 2)
        x_gps_1s[t] = dist
    x_gps_1s[0] = x_gps_1s[1] if n_sec > 1 else 0.0

    return {
        'name': name,
        'a_fwd': a_fwd_1s,
        'w_yaw': w_yaw_1s,
        'a_lat': a_lat_1s,
        'w_yaw_accel': w_yaw_accel_1s,
        'v_wheel': v_wheel_1s,
        'w_diff_yaw': w_diff_yaw_1s,
        'x_gps': x_gps_1s,
        'w_gps': w_gps_1s,
        'lats': lats_1s,
        'lons': lons_1s,
        'headings': headings_1s,
    }


# ─── Windowing ───────────────────────────────────────────────────────────────

def create_v5_windows(journeys, window_size=WINDOW_SIZE, is_train=True, noise_std=NOISE_STD):
    """
    Build (N, W, 7) temporal windows + targets.

    Channels:
      0: a_fwd (longitudinal accel)
      1: w_yaw (yaw rate)
      2: a_lat (measured lateral accel)
      3: v_prev (previous velocity feedback, noise-injected on train)
      4: w_yaw_accel (smoothed yaw acceleration)
      5: centripetal_residual (a_lat - v_prev * w_yaw)
      6: v_wheel (rear-axle linear wheel velocity, train-calibrated)
    """
    X_list = []
    y_disp_list = []
    y_ori_list = []
    y_zupt_list = []

    for j in journeys:
        a_fwd = np.clip(j['a_fwd'], *CLIP["accel"])
        w_yaw = np.clip(j['w_yaw'], *CLIP["gyro"])
        a_lat = np.clip(j['a_lat'], *CLIP["accel"])
        w_yaw_accel = np.clip(j['w_yaw_accel'], *CLIP["yaw_accel"])
        v_wheel = np.clip(j['v_wheel'], *CLIP["v_wheel"])
        x_gps = np.clip(j['x_gps'], *CLIP["disp"])
        w_gps = np.clip(j['w_gps'], *CLIP["yaw"])

        n = len(x_gps)
        if n <= window_size + 2:
            continue

        # v_prev: displacement feedback with optional training noise
        if is_train:
            v_prev = x_gps + np.random.normal(0.0, noise_std, size=n)
            v_prev = np.clip(v_prev, 0.0, CLIP["disp"][1])
        else:
            v_prev = x_gps.copy()

        # Derived centripetal residual: a_lat - v_prev * w_yaw
        centripetal_res = a_lat - v_prev * w_yaw
        centripetal_res = np.clip(centripetal_res, *CLIP["centripetal_res"])

        for t in range(window_size, n):
            # 7-channel temporal window: (W, 7)
            ch_a_fwd = a_fwd[t - window_size + 1: t + 1]
            ch_w_yaw = w_yaw[t - window_size + 1: t + 1]
            ch_a_lat = a_lat[t - window_size + 1: t + 1]
            ch_v_prev = v_prev[t - window_size: t]  # shifted by 1 (feedback from previous step)
            ch_w_accel = w_yaw_accel[t - window_size + 1: t + 1]
            ch_centripetal = centripetal_res[t - window_size + 1: t + 1]
            ch_v_wheel = v_wheel[t - window_size + 1: t + 1]

            window = np.stack([
                ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal, ch_v_wheel
            ], axis=-1)  # (10, 7)
            X_list.append(window)

            # Targets
            y_disp_list.append(x_gps[t])
            y_ori_list.append(w_gps[t])

            # ZUPT label: stationary if speed < 0.2 m/s AND |accel| < 0.1 m/s²
            is_stopped = 1.0 if (x_gps[t] < 0.2 and abs(a_fwd[t]) < 0.1) else 0.0
            y_zupt_list.append(is_stopped)

    X = np.array(X_list, dtype=np.float32)             # (N, 10, 7)
    y_disp = np.array(y_disp_list, dtype=np.float32).reshape(-1, 1)
    y_ori = np.array(y_ori_list, dtype=np.float32).reshape(-1, 1)
    y_zupt = np.array(y_zupt_list, dtype=np.float32).reshape(-1, 1)

    return X, y_disp, y_ori, y_zupt


# ─── Main pipeline ───────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process all journeys ──
    pairs = find_synchronized_pairs()
    print(f"[v5] Found {len(pairs)} synchronized (S, V) pairs.")

    all_journeys = []
    total_seconds = 0
    for key in sorted(pairs.keys()):
        s_p, v_p = pairs[key]
        j = process_journey_v5(s_p, v_p, name=key)
        if j is not None:
            all_journeys.append(j)
            total_seconds += len(j['x_gps'])

    print(f"[v5] Processed {len(all_journeys)} journeys, {total_seconds:,} seconds total.")

    # ── Trip-level split BEFORE windowing (Strictly preserving v3/v4 split, seed=42) ──
    test_named_tags = set()
    for tags in TEST_SCENARIOS.values():
        for tag in tags:
            test_named_tags.add(tag.lower().replace('-', '').replace('_', ''))

    test_journeys = []
    remaining_journeys = []
    for j in all_journeys:
        if j['name'].lower() in test_named_tags:
            test_journeys.append(j)
        else:
            remaining_journeys.append(j)

    target_test_secs = int(total_seconds * 0.20)
    target_val_secs = int(total_seconds * 0.10)

    current_test_secs = sum(len(j['x_gps']) for j in test_journeys)

    np.random.seed(42)
    indices = np.random.permutation(len(remaining_journeys))
    pool = [remaining_journeys[i] for i in indices]

    idx = 0
    while current_test_secs < target_test_secs and idx < len(pool):
        test_journeys.append(pool[idx])
        current_test_secs += len(pool[idx]['x_gps'])
        idx += 1

    val_journeys = []
    current_val_secs = 0
    while current_val_secs < target_val_secs and idx < len(pool):
        val_journeys.append(pool[idx])
        current_val_secs += len(pool[idx]['x_gps'])
        idx += 1

    train_journeys = pool[idx:]
    current_train_secs = sum(len(j['x_gps']) for j in train_journeys)
    actual_total = current_train_secs + current_val_secs + current_test_secs

    print(f"\n[v5] Frozen Split: {len(train_journeys)} train journeys / "
          f"{len(val_journeys)} val journeys / {len(test_journeys)} test journeys")
    print(f"  Train: {current_train_secs:,}s ({current_train_secs/actual_total*100:.1f}%)")
    print(f"  Val:   {current_val_secs:,}s ({current_val_secs/actual_total*100:.1f}%)")
    print(f"  Test:  {current_test_secs:,}s ({current_test_secs/actual_total*100:.1f}%)")

    # Verify zero overlap
    train_names = {j['name'] for j in train_journeys}
    val_names = {j['name'] for j in val_journeys}
    test_names = {j['name'] for j in test_journeys}
    assert train_names.isdisjoint(val_names), "Train/val journey overlap!"
    assert train_names.isdisjoint(test_names), "Train/test journey overlap!"
    assert val_names.isdisjoint(test_names), "Val/test journey overlap!"
    print("[v5] Verified: zero journey overlap across splits.")

    # ── Turn distribution statistics strictly from train journeys ──
    all_train_w_yaw = np.concatenate([j['w_yaw'] for j in train_journeys])
    all_train_w_accel = np.concatenate([j['w_yaw_accel'] for j in train_journeys])

    std_w_yaw = float(np.std(all_train_w_yaw)) + 1e-6
    std_w_accel = float(np.std(all_train_w_accel)) + 1e-6
    turn_scores_train = np.sqrt((all_train_w_yaw / std_w_yaw)**2 + (all_train_w_accel / std_w_accel)**2)
    tau_turn = float(np.percentile(turn_scores_train, 75))

    print(f"\n[v5] Calibrated Train Constants:")
    print(f"  r_tire (locked):   {R_TIRE_TRAIN:.6f} m")
    print(f"  track_width:       {TRACK_WIDTH_TRAIN:.4f} m")
    print(f"  std(w_yaw):        {std_w_yaw:.4f} rad/s")
    print(f"  std(w_yaw_accel):  {std_w_accel:.4f} rad/s²")
    print(f"  tau_turn (p75):    {tau_turn:.4f}")

    # ── Create (N, 10, 7) windows ──
    print("\n[v5] Creating 7-channel windows...")
    X_train, yd_train, yo_train, yz_train = create_v5_windows(train_journeys, is_train=True)
    X_val, yd_val, yo_val, yz_val = create_v5_windows(val_journeys, is_train=False)
    X_test, yd_test, yo_test, yz_test = create_v5_windows(test_journeys, is_train=False)

    print(f"  Train: X={X_train.shape}, y_disp={yd_train.shape}, ZUPT+={yz_train.mean():.3f}")
    print(f"  Val:   X={X_val.shape}, y_disp={yd_val.shape}, ZUPT+={yz_val.mean():.3f}")
    print(f"  Test:  X={X_test.shape}, y_disp={yd_test.shape}, ZUPT+={yz_test.mean():.3f}")

    # ── Fit scalers strictly on train split ──
    print("\n[v5] Fitting scalers strictly on train windows...")
    s_X = MinMaxScaler(feature_range=(0, 1))
    X_tr_flat = X_train.reshape(len(X_train), -1)
    s_X.fit(X_tr_flat)

    s_yd = MinMaxScaler(feature_range=(0, 1))
    s_yd.fit(yd_train)

    s_yo = MinMaxScaler(feature_range=(0, 1))
    s_yo.fit(yo_train)

    # Transform all splits
    X_tr_scaled = s_X.transform(X_tr_flat).reshape(X_train.shape).astype(np.float32)
    X_va_scaled = s_X.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape).astype(np.float32)
    X_te_scaled = s_X.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape).astype(np.float32)

    yd_tr_scaled = s_yd.transform(yd_train).astype(np.float32)
    yd_va_scaled = s_yd.transform(yd_val).astype(np.float32)
    yd_te_scaled = s_yd.transform(yd_test).astype(np.float32)

    yo_tr_scaled = s_yo.transform(yo_train).astype(np.float32)
    yo_va_scaled = s_yo.transform(yo_val).astype(np.float32)
    yo_te_scaled = s_yo.transform(yo_test).astype(np.float32)

    # ── Save datasets ──
    np.savez_compressed(
        DATA_DIR / "dataset_splits_v5.npz",
        X_train=X_tr_scaled, y_disp_train=yd_tr_scaled, y_ori_train=yo_tr_scaled, y_zupt_train=yz_train,
        X_val=X_va_scaled,   y_disp_val=yd_va_scaled,   y_ori_val=yo_va_scaled,   y_zupt_val=yz_val,
        X_test=X_te_scaled,  y_disp_test=yd_te_scaled,  y_ori_test=yo_te_scaled,  y_zupt_test=yz_test,
    )
    print(f"[v5] Saved: {DATA_DIR / 'dataset_splits_v5.npz'}")

    scalers = {
        "X": s_X,
        "y_disp": s_yd,
        "y_ori": s_yo,
        "clip": CLIP,
        "r_tire": R_TIRE_TRAIN,
        "track_width": TRACK_WIDTH_TRAIN,
        "turn_stats": {
            "std_w_yaw": std_w_yaw,
            "std_w_accel": std_w_accel,
            "tau_turn": tau_turn,
        }
    }
    with open(DATA_DIR / "scalers_v5.pkl", "wb") as f:
        pickle.dump(scalers, f)
    print(f"[v5] Saved: {DATA_DIR / 'scalers_v5.pkl'}")

    # ── Save raw test journeys for closed-loop evaluation ──
    test_dict = {j['name']: j for j in test_journeys}
    with open(DATA_DIR / "test_scenarios_v5.pkl", "wb") as f:
        pickle.dump(test_dict, f)
    print(f"[v5] Saved: {DATA_DIR / 'test_scenarios_v5.pkl'} ({len(test_dict)} journeys)")

    metadata = {
        "channels": [
            "a_fwd",
            "w_yaw",
            "a_lat_measured",
            "v_prev",
            "w_yaw_accel",
            "centripetal_residual",
            "v_wheel"
        ],
        "num_channels": 7,
        "window_size": WINDOW_SIZE,
        "noise_std": NOISE_STD,
        "r_tire": R_TIRE_TRAIN,
        "track_width": TRACK_WIDTH_TRAIN,
        "clip": CLIP,
        "sensor_source": "smartphone_IMU (S- files) + CAN-bus rear wheel speed (V- files)",
        "train_journeys": len(train_journeys),
        "val_journeys": len(val_journeys),
        "test_journeys": len(test_journeys),
        "train_windows": len(X_train),
        "val_windows": len(X_val),
        "test_windows": len(X_test),
        "turn_stats": scalers["turn_stats"],
        "split_method": "frozen trip-level before windowing, seed=42"
    }
    with open(DATA_DIR / "metadata_v5.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[v5] Saved: {DATA_DIR / 'metadata_v5.json'}")
    print("\n[v5] Preprocessing complete.")


if __name__ == "__main__":
    main()
