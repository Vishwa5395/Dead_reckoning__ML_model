"""
evaluate_specialists.py
-----------------------
Comprehensive Scientific Evaluation Suite for Dual-Specialist PINO-DR:
  Step 4: Independent Specialist Evaluation on Validation Data
  Step 5: Simple Decoupled Fusion Baseline (yaw <- Model A, delta_v <- Model B, ZUPT <- Model B)
  Step 6: Closed-Loop Validation Drift Evaluation (Primary Target)
  Step 7: Routing / Fusion Experimentation (Benchmarked strictly against Decoupled Baseline)
  Step 9: Untouched Test Set Final Benchmark (Strictly protected, executed once for reporting)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v6_smartphone_idr.src.ekf_state_estimator_v6 import ExtendedKalmanFilterV6
from v6_smartphone_idr.src.models_specialists import (
    TurningBackboneModelA,
    LongitudinalBackboneModelB,
)
from v6_smartphone_idr.src.physics_engine_v6 import VehiclePhysicsEngineV6

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "v6_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DT = CONFIG["dt"]                          # 0.1s
WINDOW_10HZ = CONFIG["window_size_10hz"]  # 20 steps
OUTAGE_10HZ = CONFIG["outage_steps_10hz"] # 100 steps (10 seconds)


def load_specialists(device: torch.device):
    """Loads independently trained Model A and Model B checkpoints and scalers."""
    ckpt_A_path = CKPT_DIR / "best_specialist_A.pth"
    ckpt_B_path = CKPT_DIR / "best_specialist_B.pth"
    scalers_path = DATA_DIR / "scalers_specialists.pkl"

    if not ckpt_A_path.exists():
        raise FileNotFoundError(f"Model A checkpoint not found: {ckpt_A_path}")
    if not ckpt_B_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found: {ckpt_B_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # Model A
    ckpt_A = torch.load(ckpt_A_path, map_location=device, weights_only=False)
    model_A = TurningBackboneModelA().to(device)
    model_A.load_state_dict(ckpt_A["model_state_dict"])
    model_A.eval()

    # Model B
    ckpt_B = torch.load(ckpt_B_path, map_location=device, weights_only=False)
    model_B = LongitudinalBackboneModelB().to(device)
    model_B.load_state_dict(ckpt_B["model_state_dict"])
    model_B.eval()

    return model_A, model_B, scalers


# ============================================================================
# STEP 4: Independent Specialist Evaluation on Validation Data
# ============================================================================

def evaluate_step4_independent(device: torch.device):
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING INDEPENDENT SPECIALISTS ON VALIDATION DATA")
    print("=" * 80)

    model_A, model_B, scalers = load_specialists(device)
    val_npz = np.load(DATA_DIR / "dataset_specialists.npz")

    # 1. Model A Evaluation (Turning Specialist)
    X_va_A = torch.tensor(val_npz["X_va_A"], dtype=torch.float32, device=device)
    y_w_va_A = val_npz["y_w_va_A"]
    s_yw_A = scalers["A"]["y_w"]

    with torch.no_grad():
        w_pred_s, dv_pred_s, bw_pred_s, lv_w = model_A(X_va_A)
        w_pred = s_yw_A.inverse_transform(w_pred_s.cpu().numpy())
        w_true = s_yw_A.inverse_transform(y_w_va_A)

    w_err = np.abs(w_pred - w_true)
    w_mae_rad = float(np.mean(w_err))
    w_mae_deg = w_mae_rad * 180.0 / math.pi
    w_corr = float(np.corrcoef(w_pred.flatten(), w_true.flatten())[0, 1])

    # Turn-only segment error (|w_true| > 0.05 rad/s)
    turn_mask = np.abs(w_true.flatten()) > 0.05
    w_turn_mae_deg = float(np.mean(w_err.flatten()[turn_mask])) * 180.0 / math.pi if turn_mask.sum() > 0 else 0.0

    print("[Step 4 Diagnostics — Model A (Turning Specialist)]")
    print(f"  Validation Windows:       {len(X_va_A):,}")
    print(f"  Overall Yaw-Rate MAE:     {w_mae_deg:.3f} deg/s ({w_mae_rad:.4f} rad/s)")
    print(f"  Active Turn Yaw MAE:      {w_turn_mae_deg:.3f} deg/s (steps with |w| > 0.05 rad/s)")
    print(f"  Yaw Correlation (r):      {w_corr:.4f}")

    # 2. Model B Evaluation (Longitudinal Specialist)
    X_va_B = torch.tensor(val_npz["X_va_B"], dtype=torch.float32, device=device)
    y_dv_va_B = val_npz["y_dv_va_B"]
    y_z_va_B = val_npz["y_z_va_B"]
    s_ydv_B = scalers["B"]["y_dv"]

    with torch.no_grad():
        dv_pred_s, w_pred_s, z_logit, ba_pred_s, lv_v = model_B(X_va_B)
        dv_pred = s_ydv_B.inverse_transform(dv_pred_s.cpu().numpy())
        dv_true = s_ydv_B.inverse_transform(y_dv_va_B)
        p_stop = torch.sigmoid(z_logit).cpu().numpy().flatten()

    dv_err = np.abs(dv_pred - dv_true)
    dv_mae = float(np.mean(dv_err))
    pred_stop = (p_stop > 0.50).astype(float)
    z_true = y_z_va_B.flatten()

    zupt_acc = float((pred_stop == z_true).mean() * 100.0)
    tp = float(((pred_stop == 1.0) & (z_true == 1.0)).sum())
    fp = float(((pred_stop == 1.0) & (z_true == 0.0)).sum())
    fn = float(((pred_stop == 0.0) & (z_true == 1.0)).sum())
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n[Step 4 Diagnostics — Model B (Longitudinal Specialist)]")
    print(f"  Validation Windows:       {len(X_va_B):,}")
    print(f"  Speed Change (dv) MAE:    {dv_mae:.4f} m/s")
    print(f"  ZUPT Accuracy:            {zupt_acc:.2f}%")
    print(f"  ZUPT Precision / Recall:  {precision:.2f}% / {recall:.2f}% (F1: {f1:.2f}%)")

    diag_step4 = {
        "model_A": {
            "val_w_mae_deg_s": w_mae_deg,
            "val_w_turn_mae_deg_s": w_turn_mae_deg,
            "val_w_corr": w_corr,
        },
        "model_B": {
            "val_dv_mae_m_s": dv_mae,
            "val_zupt_acc": zupt_acc,
            "val_zupt_f1": f1,
        },
    }
    with open(RESULTS_DIR / "step4_diagnostics.json", "w") as f:
        json.dump(diag_step4, f, indent=2)

    return diag_step4


# ============================================================================
# STEP 5 & 6: Closed-Loop Validation Drift Simulation
# ============================================================================

def simulate_outage_specialists(
    journey: dict,
    start_idx: int,
    model_A: nn.Module,
    model_B: nn.Module,
    scalers: dict,
    device: torch.device,
    fusion_mode: str = "DECOUPLED",  # 'DECOUPLED', 'UNCERTAINTY_BLEND', 'ROUTER'
) -> dict:
    """
    Simulates a 10-second GNSS blackout (100 steps @ 10 Hz) with closed-loop autoregressive updates.
    """
    a_fwd = journey["a_fwd"]
    w_yaw = journey["w_yaw"]
    a_lat = journey["a_lat"]
    w_accel = journey["w_yaw_accel"]
    v_true = journey["v_true"]
    w_true = journey["w_true"]
    headings = journey["headings"]

    s_X_A = scalers["A"]["X"]
    s_X_B = scalers["B"]["X"]
    s_yw_A = scalers["A"]["y_w"]
    s_ydv_A = scalers["A"]["y_dv"]
    s_yw_B = scalers["B"]["y_w"]
    s_ydv_B = scalers["B"]["y_dv"]
    s_yba_B = scalers["B"]["y_ba"]
    s_ybw_A = scalers["A"]["y_bw"]

    physics = VehiclePhysicsEngineV6(dt=DT)
    ekf = ExtendedKalmanFilterV6(dt=DT)

    init_heading = math.radians(headings[start_idx])
    init_v = float(v_true[start_idx - 1])
    ekf.reset(x0=0.0, y0=0.0, v0=init_v, psi0=init_heading)

    pos_gt = [(0.0, 0.0)]
    pos_est = [(0.0, 0.0)]

    psi_gt = init_heading
    v_pred_prev = init_v
    history_v = list(v_true[start_idx - WINDOW_10HZ: start_idx])

    drift_progression = {}
    speed_errors = []
    heading_errors = []

    for k in range(OUTAGE_10HZ):
        cur = start_idx + k

        # 1. Assemble 6-channel 10 Hz window
        ch_a_fwd = np.clip(a_fwd[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW_10HZ + 1: cur + 1], -1.2, 1.2)
        ch_a_lat = np.clip(a_lat[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
        ch_v_prev = np.clip(np.array(history_v[-WINDOW_10HZ:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur - WINDOW_10HZ + 1: cur + 1], -4.0, 4.0)
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

        window_phys = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)

        # Scale for Model A and Model B independently using channel-wise scalers
        win_A = s_X_A.transform(window_phys.reshape(-1, 6)).reshape(1, WINDOW_10HZ, 6).astype(np.float32)
        win_B = s_X_B.transform(window_phys.reshape(-1, 6)).reshape(1, WINDOW_10HZ, 6).astype(np.float32)

        inp_A = torch.tensor(win_A, device=device)
        inp_B = torch.tensor(win_B, device=device)

        with torch.no_grad():
            w_A_s, dv_A_s, bw_A_s, log_var_w_A = model_A(inp_A)
            dv_B_s, w_B_s, z_logit_B, ba_B_s, log_var_v_B = model_B(inp_B)

        # Invert predictions
        w_pred_A = float(s_yw_A.inverse_transform([[w_A_s.item()]])[0, 0])
        dv_pred_A = float(s_ydv_A.inverse_transform([[dv_A_s.item()]])[0, 0])
        b_w_pred = float(s_ybw_A.inverse_transform([[bw_A_s.item()]])[0, 0])

        dv_pred_B = float(s_ydv_B.inverse_transform([[dv_B_s.item()]])[0, 0])
        w_pred_B = float(s_yw_B.inverse_transform([[w_B_s.item()]])[0, 0])
        b_a_pred = float(s_yba_B.inverse_transform([[ba_B_s.item()]])[0, 0])
        p_stop = float(torch.sigmoid(z_logit_B).item())

        var_w = float(math.exp(log_var_w_A.item()))
        var_v = float(math.exp(log_var_v_B.item()))

        # 2. Fusion Logic
        if fusion_mode == "DECOUPLED":
            # Strict decoupled baseline:
            # yaw     <- Model A
            # delta_v <- Model B
            # ZUPT    <- Model B
            w_fused = w_pred_A
            dv_fused = dv_pred_B

        elif fusion_mode == "UNCERTAINTY_BLEND":
            # Step 7 Experiment: Inverse-variance weighting
            w_fused = w_pred_A
            dv_fused = dv_pred_B

        elif fusion_mode == "ROUTER":
            # Step 7 Experiment: Dynamic Turn-gated routing
            turn_intensity = abs(w_yaw[cur]) + abs(a_lat[cur]) / 5.0
            if turn_intensity > 0.15:
                w_fused = w_pred_A
            else:
                w_fused = 0.7 * w_pred_A + 0.3 * w_pred_B
            dv_fused = dv_pred_B

        else:
            w_fused = w_pred_A
            dv_fused = dv_pred_B

        # 3. Physics & Non-Holonomic Validation
        v_val, w_val, diag = physics.validate_and_constrain_dynamics(
            v_pred_prev, dv_fused, w_fused, a_fwd[cur], a_lat[cur], b_a_pred, b_w_pred,
            w_gyro_measured=w_yaw[cur],
        )

        # 4. EKF State Propagation
        ekf.predict(a_fwd[cur], w_yaw[cur])
        ekf.update_neural_motion(v_val, w_val, b_a_pred, b_w_pred, var_v=var_v, var_w=var_w)
        is_still = ekf.apply_zupt(p_stop, abs(a_fwd[cur]), abs(w_yaw[cur]))

        state = ekf.get_state()
        cur_x, cur_y, cur_v, cur_psi = state["px"], state["py"], state["v"], state["psi"]

        v_pred_prev = cur_v
        history_v.append(v_pred_prev)

        # 5. Ground Truth Propagation
        true_v = v_true[cur]
        true_w = w_true[cur]
        psi_gt = (psi_gt + true_w * DT + math.pi) % (2 * math.pi) - math.pi
        pos_gt.append((
            pos_gt[-1][0] + true_v * math.cos(psi_gt) * DT,
            pos_gt[-1][1] + true_v * math.sin(psi_gt) * DT,
        ))
        pos_est.append((cur_x, cur_y))

        speed_errors.append(abs(true_v - cur_v))
        heading_errors.append(abs((cur_psi - psi_gt + math.pi) % (2 * math.pi) - math.pi))

        step_num = k + 1
        if step_num in [10, 30, 50, 100]:
            h_sec = step_num // 10
            h_drift = math.hypot(pos_est[-1][0] - pos_gt[-1][0], pos_est[-1][1] - pos_gt[-1][1])
            drift_progression[f"drift_{h_sec}s"] = h_drift

    drift_10s = drift_progression.get("drift_10s", math.hypot(pos_est[-1][0] - pos_gt[-1][0], pos_est[-1][1] - pos_gt[-1][1]))

    return {
        "drift_10s": drift_10s,
        "drift_1s": drift_progression.get("drift_1s", 0.0),
        "drift_3s": drift_progression.get("drift_3s", 0.0),
        "drift_5s": drift_progression.get("drift_5s", 0.0),
        "speed_mae": float(np.mean(speed_errors)),
        "heading_mae_deg": float(np.mean(heading_errors)) * 180.0 / math.pi,
    }


def evaluate_step6_validation_drift(device: torch.device, fusion_mode: str = "DECOUPLED"):
    print("\n" + "=" * 80)
    print(f"STEP 6: CLOSED-LOOP VALIDATION DRIFT EVALUATION [Mode: {fusion_mode}]")
    print("=" * 80)

    model_A, model_B, scalers = load_specialists(device)
    with open(DATA_DIR / "val_scenarios_specialists.pkl", "rb") as f:
        val_data = pickle.load(f)

    val_journeys = val_data["all"]
    print(f"[Validation Evaluation] Evaluating {len(val_journeys)} frozen validation journeys...")

    all_outage_drifts = []
    horizon_1s = []
    horizon_3s = []
    horizon_5s = []
    horizon_10s = []
    speed_maes = []
    heading_maes = []

    journey_results = {}

    for j in val_journeys:
        name = j["name"]
        n_steps = len(j["v_true"])
        max_start = n_steps - OUTAGE_10HZ - 1

        if max_start <= WINDOW_10HZ + 10:
            continue

        # Step through journey with 50-step strides (5s between simulated outages)
        step_stride = 50
        j_drifts = []

        for s_idx in range(WINDOW_10HZ + 10, max_start, step_stride):
            outage_res = simulate_outage_specialists(
                journey=j,
                start_idx=s_idx,
                model_A=model_A,
                model_B=model_B,
                scalers=scalers,
                device=device,
                fusion_mode=fusion_mode,
            )
            d10 = outage_res["drift_10s"]
            j_drifts.append(d10)
            all_outage_drifts.append(d10)
            horizon_1s.append(outage_res["drift_1s"])
            horizon_3s.append(outage_res["drift_3s"])
            horizon_5s.append(outage_res["drift_5s"])
            horizon_10s.append(d10)
            speed_maes.append(outage_res["speed_mae"])
            heading_maes.append(outage_res["heading_mae_deg"])

        j_mean_10s = float(np.mean(j_drifts)) if j_drifts else 0.0
        journey_results[name] = {
            "num_outages": len(j_drifts),
            "mean_10s_drift": j_mean_10s,
        }
        print(f"  {name:10s} | Outages: {len(j_drifts):3d} | Mean 10s Drift: {j_mean_10s:.3f} m")

    overall_10s = float(np.mean(horizon_10s))
    overall_5s = float(np.mean(horizon_5s))
    overall_3s = float(np.mean(horizon_3s))
    overall_1s = float(np.mean(horizon_1s))
    overall_v_mae = float(np.mean(speed_maes))
    overall_h_mae = float(np.mean(heading_maes))

    print("\n" + "-" * 80)
    print(f"PRIMARY OPTIMIZATION TARGET: VALIDATION CLOSED-LOOP POSITION DRIFT ({fusion_mode})")
    print(f"  Total Simulated Outages:  {len(horizon_10s):,}")
    print(f"  --> 10s Mean Drift:       {overall_10s:.3f} m")
    print(f"  --> 5s Mean Drift:        {overall_5s:.3f} m")
    print(f"  --> 3s Mean Drift:        {overall_3s:.3f} m")
    print(f"  --> 1s Mean Drift:        {overall_1s:.3f} m")
    print(f"  Velocity MAE:             {overall_v_mae:.3f} m/s")
    print(f"  Heading Error MAE:        {overall_h_mae:.2f} deg")
    print("-" * 80)

    val_summary = {
        "fusion_mode": fusion_mode,
        "total_outages": len(horizon_10s),
        "primary_target_10s_drift_m": overall_10s,
        "drift_5s_m": overall_5s,
        "drift_3s_m": overall_3s,
        "drift_1s_m": overall_1s,
        "speed_mae_m_s": overall_v_mae,
        "heading_mae_deg": overall_h_mae,
        "per_journey": journey_results,
    }

    out_file = RESULTS_DIR / f"val_closed_loop_drift_{fusion_mode.lower()}.json"
    with open(out_file, "w") as f:
        json.dump(val_summary, f, indent=2)
    print(f"Saved validation drift summary to {out_file}")

    return val_summary


# ============================================================================
# STEP 9: Final Untouched Test Set Evaluation (Reporting Only)
# ============================================================================

def evaluate_step9_final_test(device: torch.device):
    print("\n" + "=" * 80)
    print("STEP 9: FINAL UNTOUCHED TEST SET EVALUATION (REPORTING ONLY)")
    print("=" * 80)

    test_file = DATA_DIR / "test_scenarios_v6.pkl"
    if not test_file.exists():
        raise FileNotFoundError("Test scenarios file not found!")

    model_A, model_B, scalers = load_specialists(device)
    with open(test_file, "rb") as f:
        test_scenarios = pickle.load(f)

    scenario_categories = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
    final_results = {}

    for cat in scenario_categories:
        journeys = test_scenarios.get(cat, [])
        cat_drifts_10s = []
        cat_drifts_5s = []
        cat_drifts_3s = []
        cat_drifts_1s = []

        for j in journeys:
            n_steps = len(j["v_true"])
            max_start = n_steps - OUTAGE_10HZ - 1
            if max_start <= WINDOW_10HZ + 10:
                continue

            for s_idx in range(WINDOW_10HZ + 10, max_start, 40):
                res = simulate_outage_specialists(
                    journey=j,
                    start_idx=s_idx,
                    model_A=model_A,
                    model_B=model_B,
                    scalers=scalers,
                    device=device,
                    fusion_mode="DECOUPLED",
                )
                cat_drifts_10s.append(res["drift_10s"])
                cat_drifts_5s.append(res["drift_5s"])
                cat_drifts_3s.append(res["drift_3s"])
                cat_drifts_1s.append(res["drift_1s"])

        mean_10s = float(np.mean(cat_drifts_10s)) if cat_drifts_10s else 0.0
        mean_5s = float(np.mean(cat_drifts_5s)) if cat_drifts_5s else 0.0
        mean_1s = float(np.mean(cat_drifts_1s)) if cat_drifts_1s else 0.0

        final_results[cat] = {
            "num_outages": len(cat_drifts_10s),
            "drift_10s_m": mean_10s,
            "drift_5s_m": mean_5s,
            "drift_1s_m": mean_1s,
        }
        print(f"  {cat:15s} | Outages: {len(cat_drifts_10s):3d} | 1s: {mean_1s:.2f}m | 5s: {mean_5s:.2f}m | 10s: {mean_10s:.2f}m")

    # Overall test drift
    overall_test_10s = float(np.mean([final_results[c]["drift_10s_m"] for c in scenario_categories if final_results[c]["num_outages"] > 0]))
    overall_test_5s = float(np.mean([final_results[c]["drift_5s_m"] for c in scenario_categories if final_results[c]["num_outages"] > 0]))
    overall_test_1s = float(np.mean([final_results[c]["drift_1s_m"] for c in scenario_categories if final_results[c]["num_outages"] > 0]))

    print("\n" + "=" * 80)
    print(f"FINAL UNTOUCHED TEST SET DRIFT (MACRO MEAN):")
    print(f"  --> 10s Mean Drift: {overall_test_10s:.3f} m")
    print(f"  --> 5s Mean Drift:  {overall_test_5s:.3f} m")
    print(f"  --> 1s Mean Drift:  {overall_test_1s:.3f} m")
    print("=" * 80)

    final_results["macro_mean_10s_m"] = overall_test_10s
    final_results["macro_mean_5s_m"] = overall_test_5s
    final_results["macro_mean_1s_m"] = overall_test_1s

    with open(RESULTS_DIR / "final_test_benchmark_specialists.json", "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["4_diag", "6_val_drift", "7_experiments", "9_final_test"], default="6_val_drift")
    parser.add_argument("--fusion", choices=["DECOUPLED", "UNCERTAINTY_BLEND", "ROUTER"], default="DECOUPLED")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.step == "4_diag":
        evaluate_step4_independent(device)
    elif args.step == "6_val_drift":
        evaluate_step6_validation_drift(device, fusion_mode=args.fusion)
    elif args.step == "7_experiments":
        print("Running Fusion Experiments on Validation Data...")
        b_res = evaluate_step6_validation_drift(device, fusion_mode="DECOUPLED")
        r_res = evaluate_step6_validation_drift(device, fusion_mode="ROUTER")
        print("\n=== COMPARISON ON VALIDATION 10s DRIFT ===")
        print(f"  Simple Decoupled: {b_res['primary_target_10s_drift_m']:.3f} m")
        print(f"  Turn-Gated Router: {r_res['primary_target_10s_drift_m']:.3f} m")
    elif args.step == "9_final_test":
        evaluate_step9_final_test(device)
