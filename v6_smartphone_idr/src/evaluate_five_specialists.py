"""
evaluate_five_specialists.py
----------------------------
Evaluation Engine for the 5-Independent-Specialist PINO-DR Architecture:
  - Loads M1, M2, M3, M4, M5 checkpoints.
  - Fuses predictions using the Continuous-Discrete Error-State FiveSpecialistEKF.
  - Evaluates closed-loop drift across simulated 10-second GPS outages (100 integration steps @ 10 Hz).
  - Measures 1s, 3s, 5s, and 10s trajectory drift progression.
  - Mode 'val': Evaluates all 9 validation journeys to benchmark drift.
  - Mode 'test': Single final reporting run on the untouched 13 test journeys.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v6_smartphone_idr.src.models_five_specialists import (
    VelocitySpecialistM1,
    YawSpecialistM2,
    AccelBiasSpecialistM3,
    GyroBiasSpecialistM4,
    UncertaintyAdapterM5,
)
from v6_smartphone_idr.src.ekf_estimator_five_specialists import FiveSpecialistEKF

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.1  # 10 Hz

T_M1 = 25
T_M2 = 20
T_M3 = 50
T_M4 = 50
T_M5 = 15
MAX_T = max(T_M1, T_M2, T_M3, T_M4, T_M5)


def load_specialist_models_and_scalers():
    """Loads all five trained models and their scalers."""
    scalers_path = DATA_DIR / "scalers_five_specialists.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    m1 = VelocitySpecialistM1().to(DEVICE)
    m2 = YawSpecialistM2().to(DEVICE)
    m3 = AccelBiasSpecialistM3().to(DEVICE)
    m4 = GyroBiasSpecialistM4().to(DEVICE)
    m5 = UncertaintyAdapterM5().to(DEVICE)

    m1.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_specialist_M1.pth", map_location=DEVICE))
    m2.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_specialist_M2.pth", map_location=DEVICE))
    m3.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_specialist_M3.pth", map_location=DEVICE))
    m4.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_specialist_M4.pth", map_location=DEVICE))
    m5.load_state_dict(torch.load(CHECKPOINTS_DIR / "best_specialist_M5.pth", map_location=DEVICE))

    m1.eval()
    m2.eval()
    m3.eval()
    m4.eval()
    m5.eval()

    return m1, m2, m3, m4, m5, scalers


def simulate_outage_5specialists(
    journey: dict,
    start_idx: int,
    outage_steps: int,
    m1, m2, m3, m4, m5,
    scalers: dict,
    use_ekf: bool = True
) -> dict:
    """
    Simulates a 10-second GPS outage (100 steps @ 10 Hz) using the 5-Specialist EKF.
    """
    n = len(journey["v_true"])
    end_idx = min(start_idx + outage_steps, n)
    actual_steps = end_idx - start_idx
    if actual_steps < outage_steps:
        return None

    # Reference Ground Truth Local Cartesian Trajectory
    lats = journey["lats"][start_idx:end_idx]
    lons = journey["lons"][start_idx:end_idx]
    headings = journey["headings"][start_idx:end_idx]
    v_true = journey["v_true"][start_idx:end_idx]

    lat0 = lats[0] * math.pi / 180.0
    lon0 = lons[0] * math.pi / 180.0
    r_earth = 6378137.0

    # Local ENU coordinates
    gt_x = (lons * math.pi / 180.0 - lon0) * r_earth * math.cos(lat0)
    gt_y = (lats * math.pi / 180.0 - lat0) * r_earth

    # Raw signals for journey
    ax = journey["ax"]
    ay = journey["ay"]
    az = journey["az"]
    gx = journey["gx"]
    gy = journey["gy"]
    gz = journey["gz"]
    uz = journey["uz"]
    a_fwd = journey["a_fwd"]
    a_lat = journey["a_lat"]
    w_yaw = journey["w_yaw"]
    w_accel = journey["w_yaw_accel"]
    jerk = journey["jerk"]

    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    # Initialize EKF
    ekf = FiveSpecialistEKF(dt=DT)
    init_psi = math.radians(headings[0])
    init_v = v_true[0]
    ekf.reset(init_px=gt_x[0], init_py=gt_y[0], init_v=init_v, init_psi=init_psi)

    pred_x = [gt_x[0]]
    pred_y = [gt_y[0]]
    pred_v_list = [init_v]
    pred_psi_list = [init_psi]

    current_v_est = init_v

    # Closed-loop integration loop
    for step in range(1, actual_steps):
        t = start_idx + step
        if t < MAX_T:
            return None

        # --- Build multi-scale windows ---
        # M1 (T=25)
        s1 = t - T_M1 + 1
        # v_prev feedback uses current estimated speed
        v_prev_win = np.full(T_M1, current_v_est)
        win_M1 = np.column_stack([
            ax[s1:t+1], ay[s1:t+1], az[s1:t+1],
            gx[s1:t+1], gy[s1:t+1], gz[s1:t+1],
            uz[s1:t+1, 0], uz[s1:t+1, 1], uz[s1:t+1, 2],
            a_fwd[s1:t+1], a_lat[s1:t+1], w_yaw[s1:t+1],
            jerk[s1:t+1], v_prev_win
        ])

        # M2 (T=20)
        s2 = t - T_M2 + 1
        win_M2 = np.column_stack([
            gx[s2:t+1], gy[s2:t+1], gz[s2:t+1],
            ax[s2:t+1], ay[s2:t+1], az[s2:t+1],
            uz[s2:t+1, 0], uz[s2:t+1, 1], uz[s2:t+1, 2],
            a_lat[s2:t+1], a_fwd[s2:t+1], w_yaw[s2:t+1]
        ])

        # M3 (T=50)
        s3 = t - T_M3 + 1
        win_M3 = np.column_stack([
            a_fwd[s3:t+1], a_norm[s3:t+1],
            uz[s3:t+1, 0], uz[s3:t+1, 1], uz[s3:t+1, 2],
            np.full(T_M3, current_v_est)
        ])

        # M4 (T=50)
        s4 = t - T_M4 + 1
        win_M4 = np.column_stack([
            w_yaw[s4:t+1], g_norm[s4:t+1], w_accel[s4:t+1],
            uz[s4:t+1, 0], uz[s4:t+1, 1], uz[s4:t+1, 2]
        ])

        # M5 (T=15)
        s5 = t - T_M5 + 1
        win_M5 = np.column_stack([
            ax[s5:t+1], ay[s5:t+1], az[s5:t+1],
            gx[s5:t+1], gy[s5:t+1], gz[s5:t+1],
            uz[s5:t+1, 0], uz[s5:t+1, 1], uz[s5:t+1, 2]
        ])

        # Scale features
        win_M1_s = scalers["scaler_M1"].transform(win_M1)
        win_M2_s = scalers["scaler_M2"].transform(win_M2)
        win_M3_s = scalers["scaler_M3"].transform(win_M3)
        win_M4_s = scalers["scaler_M4"].transform(win_M4)
        win_M5_s = scalers["scaler_M5"].transform(win_M5)

        # Inferences
        with torch.no_grad():
            t1 = torch.tensor(win_M1_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t2 = torch.tensor(win_M2_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t3 = torch.tensor(win_M3_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t4 = torch.tensor(win_M4_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t5 = torch.tensor(win_M5_s, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            pred_v_s, pred_dv_s = m1(t1)
            pred_w_s, pred_dpsi_s = m2(t2)
            pred_ba_s = m3(t3)
            pred_bg_s = m4(t4)
            sigmas = m5(t5)

        # Unscale predictions
        v_pred = float(pred_v_s[0, 0].cpu()) * scalers["scaler_y_v"].scale_[0]
        dv_pred = float(pred_dv_s[0, 0].cpu()) * scalers["scaler_y_v"].scale_[1]
        w_pred = float(pred_w_s[0, 0].cpu()) * scalers["scaler_y_w"].scale_[0]
        ba_pred = float(pred_ba_s[0, 0].cpu()) * scalers["scaler_y_ba"].scale_[0]
        bg_pred = float(pred_bg_s[0, 0].cpu()) * scalers["scaler_y_bg"].scale_[0]

        sigma_v = float(sigmas[0, 0].cpu())
        sigma_w = float(sigmas[0, 1].cpu())
        sigma_ba = float(sigmas[0, 2].cpu())
        sigma_bg = float(sigmas[0, 3].cpu())

        # Update EKF
        if use_ekf:
            ekf.predict(a_fwd[t], w_yaw[t])
            ekf.update_velocity(v_pred, sigma_v)
            ekf.update_heading_rate(w_pred, sigma_w)
            ekf.update_biases(ba_pred, bg_pred, sigma_ba, sigma_bg)

            # Physics ZUPT Check
            if abs(a_fwd[t]) < 0.20 and abs(w_yaw[t]) < 0.02 and v_pred < 0.35:
                ekf.apply_zupt()

            st = ekf.get_state()
            px_cur, py_cur, v_cur, psi_cur = st["px"], st["py"], st["v"], st["psi"]
        else:
            # Simple decoupled baseline: integrate M1 speed + M2 yaw rate
            px_prev, py_prev = pred_x[-1], pred_y[-1]
            psi_prev = pred_psi_list[-1]
            psi_cur = psi_prev + w_pred * DT
            v_cur = max(0.0, v_pred)
            if abs(a_fwd[t]) < 0.20 and abs(w_yaw[t]) < 0.02 and v_cur < 0.35:
                v_cur = 0.0
            px_cur = px_prev + v_cur * math.cos(psi_cur) * DT
            py_cur = py_prev + v_cur * math.sin(psi_cur) * DT

        current_v_est = v_cur
        pred_x.append(px_cur)
        pred_y.append(py_cur)
        pred_v_list.append(v_cur)
        pred_psi_list.append(psi_cur)

    # Compute Euclidean drift across horizon
    drift_series = np.sqrt((np.array(pred_x) - gt_x)**2 + (np.array(pred_y) - gt_y)**2)

    return {
        "drift_1s": float(drift_series[min(10, actual_steps - 1)]),
        "drift_3s": float(drift_series[min(30, actual_steps - 1)]),
        "drift_5s": float(drift_series[min(50, actual_steps - 1)]),
        "drift_10s": float(drift_series[-1]),
        "speed_mae": float(np.mean(np.abs(np.array(pred_v_list) - v_true))),
    }


def run_validation_drift_benchmark():
    """Runs closed-loop 10-second GPS outage drift evaluation on all 9 validation journeys."""
    print("=" * 80)
    print("  PINO-DR: Closed-Loop Validation Benchmark (9 Journeys, 10s Outages)")
    print("=" * 80)

    val_meta_path = DATA_DIR / "val_journeys_five_specialists.pkl"
    if not val_meta_path.exists():
        print(f"[Error] {val_meta_path} not found.")
        sys.exit(1)

    with open(val_meta_path, "rb") as f:
        val_journeys = pickle.load(f)

    m1, m2, m3, m4, m5, scalers = load_specialist_models_and_scalers()

    outage_length_steps = 100  # 10.0 seconds @ 10 Hz
    stride = 50                # Outage every 5.0 seconds

    # Benchmark both Full 5-Specialist EKF and Decoupled Baseline
    for mode_name, use_ekf in [("5-Specialist EKF", True), ("Decoupled M1+M2 Baseline", False)]:
        print(f"\nEvaluating Configuration: [{mode_name}]...")
        all_d1, all_d3, all_d5, all_d10 = [], [], [], []
        speed_maes = []
        total_outages = 0

        for j in val_journeys:
            n = len(j["v_true"])
            for start in range(MAX_T + 10, n - outage_length_steps - 5, stride):
                res = simulate_outage_5specialists(
                    j, start, outage_length_steps, m1, m2, m3, m4, m5, scalers, use_ekf=use_ekf
                )
                if res is not None:
                    all_d1.append(res["drift_1s"])
                    all_d3.append(res["drift_3s"])
                    all_d5.append(res["drift_5s"])
                    all_d10.append(res["drift_10s"])
                    speed_maes.append(res["speed_mae"])
                    total_outages += 1

        d1_mean = float(np.mean(all_d1))
        d3_mean = float(np.mean(all_d3))
        d5_mean = float(np.mean(all_d5))
        d10_mean = float(np.mean(all_d10))
        speed_mae_mean = float(np.mean(speed_maes))

        print(f"  Total Simulated Outages: {total_outages}")
        print(f"  1-Second Drift:  {d1_mean:.3f} m")
        print(f"  3-Second Drift:  {d3_mean:.3f} m")
        print(f"  5-Second Drift:  {d5_mean:.3f} m")
        print(f"  10-Second Drift: {d10_mean:.3f} m (PRIMARY OBJECTIVE)")
        print(f"  Speed MAE:       {speed_mae_mean:.3f} m/s")

        if use_ekf:
            val_results = {
                "configuration": "FiveSpecialistEKF",
                "total_outages": total_outages,
                "drift_1s_m": d1_mean,
                "drift_3s_m": d3_mean,
                "drift_5s_m": d5_mean,
                "drift_10s_m": d10_mean,
                "speed_mae_m_s": speed_mae_mean,
            }
            with open(RESULTS_DIR / "val_closed_loop_drift_five_specialists.json", "w") as f:
                json.dump(val_results, f, indent=2)

    print(f"\n[Val Benchmark] Saved validation summary to {RESULTS_DIR / 'val_closed_loop_drift_five_specialists.json'}")


def run_frozen_test_benchmark():
    """Runs single reporting evaluation on the 13 frozen test journeys."""
    print("=" * 80)
    print("  PINO-DR: SINGLE FINAL REPORTING RUN ON FROZEN TEST SET (13 Journeys)")
    print("=" * 80)

    test_meta_path = DATA_DIR / "test_scenarios_v6.pkl"
    if not test_meta_path.exists():
        print(f"[Error] {test_meta_path} not found.")
        sys.exit(1)

    with open(test_meta_path, "rb") as f:
        test_scenarios = pickle.load(f)

    m1, m2, m3, m4, m5, scalers = load_specialist_models_and_scalers()

    outage_steps = 100
    stride = 30
    scenario_metrics = {}

    for scenario_name, journeys in test_scenarios.items():
        all_d1, all_d5, all_d10 = [], [], []
        outage_count = 0

        for j in journeys:
            n = len(j["v_true"])
            for start in range(MAX_T + 10, n - outage_steps - 5, stride):
                res = simulate_outage_5specialists(
                    j, start, outage_steps, m1, m2, m3, m4, m5, scalers, use_ekf=True
                )
                if res is not None:
                    all_d1.append(res["drift_1s"])
                    all_d5.append(res["drift_5s"])
                    all_d10.append(res["drift_10s"])
                    outage_count += 1

        if outage_count > 0:
            scenario_metrics[scenario_name] = {
                "num_outages": outage_count,
                "drift_1s_m": float(np.mean(all_d1)),
                "drift_5s_m": float(np.mean(all_d5)),
                "drift_10s_m": float(np.mean(all_d10)),
            }
            print(f"  [{scenario_name:15s}] Outages: {outage_count:3d} | "
                  f"1s: {np.mean(all_d1):.2f}m | 5s: {np.mean(all_d5):.2f}m | 10s: {np.mean(all_d10):.2f}m")

    macro_1s = float(np.mean([m["drift_1s_m"] for m in scenario_metrics.values()]))
    macro_5s = float(np.mean([m["drift_5s_m"] for m in scenario_metrics.values()]))
    macro_10s = float(np.mean([m["drift_10s_m"] for m in scenario_metrics.values()]))

    scenario_metrics["macro_mean_1s_m"] = macro_1s
    scenario_metrics["macro_mean_5s_m"] = macro_5s
    scenario_metrics["macro_mean_10s_m"] = macro_10s

    print("-" * 80)
    print(f"  [MACRO MEAN]     1s: {macro_1s:.2f}m | 5s: {macro_5s:.2f}m | 10s: {macro_10s:.2f}m")
    print("=" * 80)

    out_file = RESULTS_DIR / "final_test_benchmark_five_specialists.json"
    with open(out_file, "w") as f:
        json.dump(scenario_metrics, f, indent=2)
    print(f"[Test Benchmark] Successfully saved test benchmark to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_frozen_test_benchmark()
    else:
        run_validation_drift_benchmark()
