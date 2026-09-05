"""
preprocess_v3.py
----------------
Production-grade data preparation for PINO-DR v3.

Key improvements over v2:
- Enriched 4-channel temporal tensor: [a_fwd, w_yaw, a_lat_measured, v_prev]
- Raw lateral acceleration from smartphone accelerometer (not derived v*w)
- Binary ZUPT labels for auxiliary stationary classification
- Trip-level split BEFORE windowing (zero temporal leakage)
- Gaussian noise injection into v_prev during training only
- Physical clipping + MinMaxScaler fitted strictly on train
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─── Project paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DST_CACHE = ROOT / "data" / "preprocessed" / "v3"

DATASET_BASE_DIR = r"C:\Users\tiwar\OneDrive\Desktop\DATASETS\IO-VNBD\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

# Named test scenario journey tags (from the Onyekpe et al. paper)
TEST_SCENARIOS = {
    'motorway': ['Vw12'],
    'roundabout': ['Vta11'],
    'quick_accel': ['Vta12'],
    'hard_brake': ['Vw16b', 'Vw17', 'Vta9'],
    'sharp_turns': ['Vw6', 'Vw7', 'Vw8'],
}

# Physical clip thresholds (road vehicle limits — not data-derived, no leakage)
CLIP = {
    "accel": (-8.0, 8.0),       # m/s²
    "disp": (0.0, 45.0),        # m per 1-second step (equiv 0–162 km/h)
    "gyro": (-1.0, 1.0),        # rad/s
    "yaw": (-1.2, 1.2),         # rad/s
}

WINDOW_SIZE = 10
NOISE_STD = 0.5   # Gaussian noise σ for v_prev during training


# ─── Vincenty geodesic distance ──────────────────────────────────────────────

import math

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


def process_journey_v3(s_path, v_path, name="journey"):
    """
    Processes one synchronized journey into 1-second intervals.
    Returns enriched arrays including lateral acceleration.
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

    # ── Lateral acceleration: perpendicular to forward in horizontal plane ──
    # Mean uz for cross product (stable over journey)
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

    # ── Downsample 10 Hz → 1 Hz ──
    n_sec = min_len // 10
    if n_sec < 12:
        return None

    a_fwd_1s = a_longitudinal[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_yaw_1s = w_yaw_aligned[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    a_lat_1s = a_lateral[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
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
        'x_gps': x_gps_1s,
        'w_gps': w_gps_1s,
        'lats': lats_1s,
        'lons': lons_1s,
        'headings': headings_1s,
    }


# ─── Windowing ───────────────────────────────────────────────────────────────

def create_v3_windows(journeys, window_size=WINDOW_SIZE, is_train=True, noise_std=NOISE_STD):
    """
    Build (N, W, 4) temporal windows + displacement/orientation/ZUPT targets.

    Channels: [a_fwd, w_yaw, a_lat_measured, v_prev]
    v_prev gets Gaussian noise during training only.
    """
    X_list = []
    y_disp_list = []
    y_ori_list = []
    y_zupt_list = []

    for j in journeys:
        a_fwd = np.clip(j['a_fwd'], *CLIP["accel"])
        w_yaw = np.clip(j['w_yaw'], *CLIP["gyro"])
        a_lat = np.clip(j['a_lat'], *CLIP["accel"])
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

        for t in range(window_size, n):
            # 4-channel temporal window: (W, 4)
            ch_a_fwd = a_fwd[t - window_size + 1: t + 1]
            ch_w_yaw = w_yaw[t - window_size + 1: t + 1]
            ch_a_lat = a_lat[t - window_size + 1: t + 1]
            ch_v_prev = v_prev[t - window_size: t]  # shifted by 1 (previous step feedback)

            window = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev], axis=-1)  # (10, 4)
            X_list.append(window)

            # Targets
            y_disp_list.append(x_gps[t])
            y_ori_list.append(w_gps[t])

            # ZUPT label: stationary if speed < 0.2 m/s AND |accel| < 0.1 m/s²
            is_stopped = 1.0 if (x_gps[t] < 0.2 and abs(a_fwd[t]) < 0.1) else 0.0
            y_zupt_list.append(is_stopped)

    X = np.array(X_list, dtype=np.float32)             # (N, 10, 4)
    y_disp = np.array(y_disp_list, dtype=np.float32).reshape(-1, 1)
    y_ori = np.array(y_ori_list, dtype=np.float32).reshape(-1, 1)
    y_zupt = np.array(y_zupt_list, dtype=np.float32).reshape(-1, 1)

    return X, y_disp, y_ori, y_zupt


# ─── Main pipeline ───────────────────────────────────────────────────────────

def main():
    DST_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Process all journeys ──
    pairs = find_synchronized_pairs()
    print(f"[v3] Found {len(pairs)} synchronized (S, V) pairs.")

    all_journeys = []
    total_seconds = 0
    for key in sorted(pairs.keys()):
        s_p, v_p = pairs[key]
        j = process_journey_v3(s_p, v_p, name=key)
        if j is not None:
            all_journeys.append(j)
            total_seconds += len(j['x_gps'])

    print(f"[v3] Processed {len(all_journeys)} journeys, {total_seconds:,} seconds total.")

    # ── Trip-level split BEFORE windowing ──
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

    print(f"\n[v3] Split: {len(train_journeys)} train journeys / "
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
    print("[v3] Verified: zero journey overlap across splits.")

    # ── Generate windows AFTER split ──
    print("\n[v3] Windowing AFTER split (W=10, 4 channels)...")
    X_tr, y_d_tr, y_o_tr, y_z_tr = create_v3_windows(train_journeys, is_train=True)
    X_va, y_d_va, y_o_va, y_z_va = create_v3_windows(val_journeys, is_train=False)
    X_te, y_d_te, y_o_te, y_z_te = create_v3_windows(test_journeys, is_train=False)

    print(f"  Train: {X_tr.shape[0]:,} windows")
    print(f"  Val:   {X_va.shape[0]:,} windows")
    print(f"  Test:  {X_te.shape[0]:,} windows")
    print(f"  ZUPT+ in train: {y_z_tr.sum():.0f}/{len(y_z_tr)} "
          f"({y_z_tr.mean()*100:.1f}%)")

    # ── Fit scalers on train only ──
    # Per-channel scaling for the (N, 10, 4) tensor
    n_tr = X_tr.shape[0]
    X_tr_flat = X_tr.reshape(n_tr, -1)  # (N, 40)
    scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X_tr_flat)
    scaler_y_disp = MinMaxScaler(feature_range=(0, 1)).fit(y_d_tr)
    scaler_y_ori = MinMaxScaler(feature_range=(0, 1)).fit(y_o_tr)

    X_tr_s = scaler_X.transform(X_tr_flat).reshape(X_tr.shape).astype(np.float32)
    X_va_s = scaler_X.transform(X_va.reshape(X_va.shape[0], -1)).reshape(X_va.shape).astype(np.float32)
    X_te_s = scaler_X.transform(X_te.reshape(X_te.shape[0], -1)).reshape(X_te.shape).astype(np.float32)

    y_d_tr_s = scaler_y_disp.transform(y_d_tr).astype(np.float32)
    y_d_va_s = scaler_y_disp.transform(y_d_va).astype(np.float32)
    y_d_te_s = scaler_y_disp.transform(y_d_te).astype(np.float32)

    y_o_tr_s = scaler_y_ori.transform(y_o_tr).astype(np.float32)
    y_o_va_s = scaler_y_ori.transform(y_o_va).astype(np.float32)
    y_o_te_s = scaler_y_ori.transform(y_o_te).astype(np.float32)

    # ── Save ──
    scalers = {
        'X': scaler_X,
        'y_disp': scaler_y_disp,
        'y_ori': scaler_y_ori,
    }
    with open(DST_CACHE / "scalers_v3.pkl", "wb") as f:
        pickle.dump(scalers, f)

    np.savez_compressed(
        DST_CACHE / "dataset_splits_v3.npz",
        X_tr=X_tr_s, y_d_tr=y_d_tr_s, y_o_tr=y_o_tr_s, y_z_tr=y_z_tr,
        X_va=X_va_s, y_d_va=y_d_va_s, y_o_va=y_o_va_s, y_z_va=y_z_va,
        X_te=X_te_s, y_d_te=y_d_te_s, y_o_te=y_o_te_s, y_z_te=y_z_te,
    )

    # Save test scenarios for closed-loop evaluation
    test_scenario_journeys = {}
    for scen_name, tags in TEST_SCENARIOS.items():
        scen_list = []
        for tag in tags:
            tag_clean = tag.lower().replace('-', '').replace('_', '')
            for j in all_journeys:
                if j['name'].lower() == tag_clean:
                    scen_list.append(j)
                    break
        test_scenario_journeys[scen_name] = scen_list

    with open(DST_CACHE / "test_scenarios_v3.pkl", "wb") as f:
        pickle.dump(test_scenario_journeys, f)

    meta = {
        "channels": ["a_fwd", "w_yaw", "a_lat_measured", "v_prev"],
        "window_size": WINDOW_SIZE,
        "noise_std": NOISE_STD,
        "clip": CLIP,
        "sensor_source": "smartphone_IMU (S- files)",
        "train_journeys": len(train_journeys),
        "val_journeys": len(val_journeys),
        "test_journeys": len(test_journeys),
        "train_windows": int(X_tr.shape[0]),
        "val_windows": int(X_va.shape[0]),
        "test_windows": int(X_te.shape[0]),
        "zupt_pos_rate_train": float(y_z_tr.mean()),
        "split_method": "trip-level before windowing, seed=42",
    }
    with open(DST_CACHE / "metadata_v3.json", "w") as f:
        json.dump(meta, f, indent=2)

    sz = (DST_CACHE / "dataset_splits_v3.npz").stat().st_size / 1024
    print(f"\n[v3] Saved to {DST_CACHE}")
    print(f"     dataset_splits_v3.npz ({sz:.1f} KB)")
    print(f"     scalers_v3.pkl, test_scenarios_v3.pkl, metadata_v3.json")
    print("[v3] Preprocessing complete.")


if __name__ == "__main__":
    main()
