"""
dataset_loader.py
-----------------
Preprocesses the COMPLETE IO-VNBD synchronized dataset (all 72 journeys, >1.07 million rows).
Split strictly configured as 70% Train, 10% Validation, 20% Test.

Key specifications:
1. Model Inputs: S- smartphone IMU (Accel X/Y/Z, Gravity X/Y/Z, Gyro Yaw/Pitch/Roll).
2. Ground Truth: V- vehicle CAN-bus/GPS (Vincenty geodesic displacement, heading/yaw rate).
3. Leveling & NHC alignment: Gravity projection and horizontal forward alignment.
4. Sliding window: 10 time steps (W=10) with delay taps.
5. Injected Gaussian noise for training feedback simulation.
6. MinMaxScaler standardization [0, 1] fitted strictly on the 70% training split.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vincenty import vincenty

DATASET_BASE_DIR = r"C:\Users\tiwar\OneDrive\Desktop\DATASETS\IO-VNBD\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
CACHE_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\data\preprocessed"

# Named test scenarios from the paper for scenario-specific evaluation
TEST_SCENARIOS = {
    'motorway': ['Vw12'],
    'roundabout': ['Vta11'],
    'quick_accel': ['Vta12'],
    'hard_brake': ['Vw16b', 'Vw17', 'Vta9'],
    'sharp_turns': ['Vw6', 'Vw7', 'Vw8']
}


def find_synchronized_pairs(base_dir=DATASET_BASE_DIR):
    """
    Finds all synchronized (S-*.csv, V-*.csv) file pairs in the dataset.
    """
    s_files = {}
    v_files = {}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.csv'):
                continue
            full_path = os.path.join(root, f)
            fl = f.lower()
            if fl.startswith('s-'):
                key = fl.replace('s-', '').replace('.csv', '').replace('-', '').replace('_', '')
                s_files[key] = full_path
            elif fl.startswith('v-'):
                key = fl.replace('v-', '').replace('.csv', '').replace('-', '').replace('_', '')
                v_files[key] = full_path

    paired = {}
    for key, s_p in s_files.items():
        if key in v_files:
            paired[key] = (s_p, v_files[key])
    return paired


def level_and_align_smartphone_imu(df_s, df_v):
    """
    Converts smartphone raw 3-axis measurements to vehicle body frame:
    1. Leveling via gravity vector:
       uz = g / ||g|| (unit vertical downwards)
       Linear accel: a_lin = a_raw - g
       Horizontal accel: a_horiz = a_lin - (a_lin . uz) * uz
    2. Vertical Yaw Rate:
       w_vert = w_gyro . uz (rotation component around true vertical gravity axis)
    3. Longitudinal alignment (Forward direction in horizontal plane):
       Estimates forward vector u_fwd in horizontal plane maximizing forward covariance
       with vehicle acceleration (or default [1, 0, 0] under NHC).
    """
    ax = df_s.iloc[:, 9].values.astype(np.float64)
    ay = df_s.iloc[:, 10].values.astype(np.float64)
    az = df_s.iloc[:, 11].values.astype(np.float64)

    gx = df_s.iloc[:, 12].values.astype(np.float64)
    gy = df_s.iloc[:, 13].values.astype(np.float64)
    gz = df_s.iloc[:, 14].values.astype(np.float64)

    wx = df_s.iloc[:, 15].values.astype(np.float64)
    wy = df_s.iloc[:, 16].values.astype(np.float64)
    wz = df_s.iloc[:, 17].values.astype(np.float64)

    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    g_norm = np.where(g_norm < 1e-3, 9.80665, g_norm)
    uz = np.column_stack([gx / g_norm, gy / g_norm, gz / g_norm])

    a_lin = np.column_stack([ax - gx, ay - gy, az - gz])
    a_vert_mag = np.sum(a_lin * uz, axis=1, keepdims=True)
    a_horiz = a_lin - a_vert_mag * uz

    w_gyro = np.column_stack([wx, wy, wz])
    w_vert = np.sum(w_gyro * uz, axis=1)

    df_v.columns = [c.strip() for c in df_v.columns]
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

    corr = np.corrcoef(w_vert, v_yaw_rate)[0, 1] if len(w_vert) > 1 else 1.0
    yaw_sign = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
    w_yaw_aligned = yaw_sign * w_vert

    return a_longitudinal, w_yaw_aligned


def process_journey_csv(s_path, v_path, name="journey"):
    """
    Processes one synchronized journey pair (10 Hz) into 1-second navigation intervals (1 Hz).
    """
    df_s = pd.read_csv(s_path, encoding='latin1')
    df_v = pd.read_csv(v_path)
    df_v.columns = [c.strip() for c in df_v.columns]

    min_len = min(len(df_s), len(df_v))
    if min_len < 30:
        return None

    df_s = df_s.iloc[:min_len]
    df_v = df_v.iloc[:min_len]

    a_ins_10hz, w_ins_10hz = level_and_align_smartphone_imu(df_s, df_v)

    lats_10hz = df_v['Latitude (degrees)'].values.astype(np.float64)
    lons_10hz = df_v['Longitude (degrees)'].values.astype(np.float64)
    heading_10hz = df_v['Heading (degrees)'].values.astype(np.float64)
    v_yaw_10hz = np.radians(df_v['Yaw Rate (deg/sec)'].values.astype(np.float64))

    n_sec = min_len // 10
    if n_sec < 12:
        return None

    a_ins_1s = a_ins_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_ins_1s = w_ins_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)
    w_gps_1s = v_yaw_10hz[:n_sec * 10].reshape(n_sec, 10).mean(axis=1)

    lats_1s = lats_10hz[np.arange(n_sec) * 10]
    lons_1s = lons_10hz[np.arange(n_sec) * 10]
    headings_1s = heading_10hz[np.arange(n_sec) * 10]

    x_gps_1s = np.zeros(n_sec, dtype=np.float64)
    for t in range(1, n_sec):
        p1 = (lats_1s[t - 1], lons_1s[t - 1])
        p2 = (lats_1s[t], lons_1s[t])
        dist = vincenty(p1, p2)
        if dist is None or np.isnan(dist):
            dlat = (p2[0] - p1[0]) * 111139.0
            dlon = (p2[1] - p1[1]) * 111139.0 * np.cos(np.radians(p1[0]))
            dist = np.sqrt(dlat**2 + dlon**2)
        else:
            dist = dist * 1000.0
        x_gps_1s[t] = dist
    x_gps_1s[0] = x_gps_1s[1] if n_sec > 1 else 0.0

    return {
        'name': name,
        'a_ins': a_ins_1s,
        'w_ins': w_ins_1s,
        'x_gps': x_gps_1s,
        'w_gps': w_gps_1s,
        'lats': lats_1s,
        'lons': lons_1s,
        'headings': headings_1s
    }


def create_idnn_sliding_windows(journeys, window_size=10, is_train=True, noise_std=0.5):
    """
    Constructs IDNN delayed input windows (10 time steps).
    """
    disp_X_list = []
    disp_Y_list = []
    ori_X_list = []
    ori_Y_list = []

    for item in journeys:
        a_ins = item['a_ins']
        w_ins = item['w_ins']
        x_gps = item['x_gps']
        w_gps = item['w_gps']
        n = len(x_gps)
        if n <= window_size + 2:
            continue

        if is_train:
            x_feedback = x_gps + np.random.normal(0.0, noise_std, size=n)
            x_feedback = np.clip(x_feedback, 0.0, None)
        else:
            x_feedback = x_gps

        for t in range(window_size, n):
            a_win = a_ins[t - window_size + 1 : t + 1]
            x_win = x_feedback[t - window_size : t]

            disp_feat = np.concatenate([a_win, x_win])
            disp_target = x_gps[t]

            ori_feat = w_ins[t - window_size + 1 : t + 1]
            ori_target = w_gps[t]

            disp_X_list.append(disp_feat)
            disp_Y_list.append(disp_target)
            ori_X_list.append(ori_feat)
            ori_Y_list.append(ori_target)

    disp_X = np.array(disp_X_list, dtype=np.float32)
    disp_Y = np.array(disp_Y_list, dtype=np.float32).reshape(-1, 1)
    ori_X = np.array(ori_X_list, dtype=np.float32)
    ori_Y = np.array(ori_Y_list, dtype=np.float32).reshape(-1, 1)

    return disp_X, disp_Y, ori_X, ori_Y


def prepare_and_cache_dataset(cache_dir=CACHE_DIR, train_ratio=0.70, test_ratio=0.20, val_ratio=0.10):
    """
    Processes the ENTIRE synchronized dataset across all 72 journeys.
    Splits data according to train:val:test = 70:10:20.
    Standardizes features using MinMaxScaler fitted ONLY on the 70% training split.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pairs = find_synchronized_pairs()
    print(f"[DataLoader] Found {len(pairs)} synchronized (S, V) pairs in IO-VNBD.")

    # Process all 72 journeys
    all_journeys = []
    total_seconds = 0
    for key in sorted(pairs.keys()):
        s_p, v_p = pairs[key]
        j = process_journey_csv(s_p, v_p, name=key)
        if j is not None:
            all_journeys.append(j)
            total_seconds += len(j['x_gps'])

    print(f"[DataLoader] Successfully processed ALL {len(all_journeys)} journeys.")
    print(f"[DataLoader] Total 1-second navigation samples: {total_seconds:,} (~{total_seconds*10:,} raw 10Hz readings).")

    # Group journeys to ensure named test scenarios are part of the test split
    test_named_tags = {'vw12', 'vta11', 'vta12', 'vw16b', 'vw17', 'vta9', 'vw6', 'vw7', 'vw8'}
    
    # Stratified journey partitioning for zero temporal leakage:
    test_journeys = []
    remaining_journeys = []
    for j in all_journeys:
        if j['name'].lower() in test_named_tags:
            test_journeys.append(j)
        else:
            remaining_journeys.append(j)

    # Calculate target seconds for 70 / 10 / 20 split
    target_test_secs = int(total_seconds * test_ratio)
    target_val_secs = int(total_seconds * val_ratio)

    current_test_secs = sum(len(j['x_gps']) for j in test_journeys)
    
    # Fill test set up to ~20%
    np.random.seed(42)
    indices = np.random.permutation(len(remaining_journeys))
    pool = [remaining_journeys[i] for i in indices]

    idx = 0
    while current_test_secs < target_test_secs and idx < len(pool):
        test_journeys.append(pool[idx])
        current_test_secs += len(pool[idx]['x_gps'])
        idx += 1

    # Fill validation set up to ~10%
    val_journeys = []
    current_val_secs = 0
    while current_val_secs < target_val_secs and idx < len(pool):
        val_journeys.append(pool[idx])
        current_val_secs += len(pool[idx]['x_gps'])
        idx += 1

    # Remaining journeys form the 70% training set
    train_journeys = pool[idx:]
    current_train_secs = sum(len(j['x_gps']) for j in train_journeys)

    actual_total = current_train_secs + current_val_secs + current_test_secs
    print(f"[DataLoader] Dataset Split Summary (70 / 10 / 20):")
    print(f"  Train: {len(train_journeys)} journeys, {current_train_secs:,} seconds ({current_train_secs/actual_total*100:.1f}%)")
    print(f"  Val:   {len(val_journeys)} journeys, {current_val_secs:,} seconds ({current_val_secs/actual_total*100:.1f}%)")
    print(f"  Test:  {len(test_journeys)} journeys, {current_test_secs:,} seconds ({current_test_secs/actual_total*100:.1f}%)")

    # Generate sliding windows
    print("[DataLoader] Generating IDNN sliding windows (W=10)...")
    d_X_tr, d_Y_tr, o_X_tr, o_Y_tr = create_idnn_sliding_windows(train_journeys, window_size=10, is_train=True)
    d_X_va, d_Y_va, o_X_va, o_Y_va = create_idnn_sliding_windows(val_journeys, window_size=10, is_train=False)
    d_X_te, d_Y_te, o_X_te, o_Y_te = create_idnn_sliding_windows(test_journeys, window_size=10, is_train=False)

    print(f"  Train windows: Disp={d_X_tr.shape}, Ori={o_X_tr.shape}")
    print(f"  Val windows:   Disp={d_X_va.shape}, Ori={o_X_va.shape}")
    print(f"  Test windows:  Disp={d_X_te.shape}, Ori={o_X_te.shape}")

    # MinMaxScaler fitted strictly on the 70% Train split
    scaler_disp_X = MinMaxScaler(feature_range=(0, 1))
    scaler_disp_Y = MinMaxScaler(feature_range=(0, 1))
    scaler_ori_X = MinMaxScaler(feature_range=(0, 1))
    scaler_ori_Y = MinMaxScaler(feature_range=(0, 1))

    d_X_tr_scaled = scaler_disp_X.fit_transform(d_X_tr)
    d_Y_tr_scaled = scaler_disp_Y.fit_transform(d_Y_tr)
    o_X_tr_scaled = scaler_ori_X.fit_transform(o_X_tr)
    o_Y_tr_scaled = scaler_ori_Y.fit_transform(o_Y_tr)

    d_X_va_scaled = scaler_disp_X.transform(d_X_va)
    d_Y_va_scaled = scaler_disp_Y.transform(d_Y_va)
    o_X_va_scaled = scaler_ori_X.transform(o_X_va)
    o_Y_va_scaled = scaler_ori_Y.transform(o_Y_va)

    d_X_te_scaled = scaler_disp_X.transform(d_X_te)
    d_Y_te_scaled = scaler_disp_Y.transform(d_Y_te)
    o_X_te_scaled = scaler_ori_X.transform(o_X_te)
    o_Y_te_scaled = scaler_ori_Y.transform(o_Y_te)

    # Save scalers
    scalers = {
        'disp_X': scaler_disp_X,
        'disp_Y': scaler_disp_Y,
        'ori_X': scaler_ori_X,
        'ori_Y': scaler_ori_Y
    }
    with open(os.path.join(cache_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    # Save splits
    np.savez_compressed(
        os.path.join(cache_dir, 'dataset_splits.npz'),
        d_X_tr=d_X_tr_scaled, d_Y_tr=d_Y_tr_scaled,
        o_X_tr=o_X_tr_scaled, o_Y_tr=o_Y_tr_scaled,
        d_X_va=d_X_va_scaled, d_Y_va=d_Y_va_scaled,
        o_X_va=o_X_va_scaled, o_Y_va=o_Y_va_scaled,
        d_X_te=d_X_te_scaled, d_Y_te=d_Y_te_scaled,
        o_X_te=o_X_te_scaled, o_Y_te=o_Y_te_scaled,
    )

    # Also extract and save test scenarios for closed-loop outage evaluation
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

    with open(os.path.join(cache_dir, 'test_scenarios.pkl'), 'wb') as f:
        pickle.dump(test_scenario_journeys, f)

    print(f"[DataLoader] Preprocessing complete! Cached files saved in {cache_dir}")
    return scalers, test_scenario_journeys


if __name__ == '__main__':
    prepare_and_cache_dataset()
