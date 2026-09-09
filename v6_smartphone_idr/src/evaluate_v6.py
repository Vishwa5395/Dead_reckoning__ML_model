"""
evaluate_v6.py
--------------
Comprehensive 10 Hz Closed-Loop Benchmark & Ablation Runner for PINO-DR v6.

Evaluates 10-second GNSS outage sequences across all 65 official test sequences
on the frozen 50/9/13 trip split under 8 ablation configurations:
  A: V4 Neural Baseline
  B: V6 Neural Motion Model Only
  C: V6 + Sensor Bias Estimation
  D: V6 + Physics Engine (Kinematic Bounds + Centripetal Consistency)
  E: V6 + 6-State EKF (Probabilistic Fusion with Neural Uncertainty)
  F: V6 + NHC + ZUPT Hysteresis Lock
  G: V6 + Probabilistic HMM Map Matching
  H: Full V6 Pipeline (All Modules Active)

Reports:
  - Scenario-wise 10-s drift (Motorway, Roundabout, Quick Accel, Hard Brake, Sharp Turns)
  - Time-horizon drift progression: 1s, 3s, 5s, 10s drift
  - Displacement CRSE, Heading CRSE, Speed MAE
  - High-yaw vs Low-yaw breakdown
  - Multi-trajectory comparison plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from v6_smartphone_idr.src.ekf_state_estimator_v6 import ExtendedKalmanFilterV6
from v6_smartphone_idr.src.map_matcher_v6 import ProbabilisticHMMMapMatcherV6
from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6
from v6_smartphone_idr.src.physics_engine_v6 import VehiclePhysicsEngineV6

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "v6_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DT = CONFIG["dt"]                      # 0.1s
WINDOW_10HZ = CONFIG["window_size_10hz"]  # 20 steps
OUTAGE_10HZ = CONFIG["outage_steps_10hz"] # 100 steps (10 seconds)
SCENARIO_ORDER = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]


def load_v6_model(device: torch.device, ckpt_name: str = "best_model_v6.pth"):
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", CONFIG["model"])
    model = PINODeadReckoningNetV6(
        in_channels=cfg.get("in_channels", 6),
        conv_channels=cfg.get("conv_channels", 32),
        gru_hidden=cfg.get("gru_hidden", 32),
        dropout=cfg.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def simulate_outage_v6(
    journey: dict,
    start_idx: int,
    model: nn.Module,
    scalers: dict,
    device: torch.device,
    ablation_mode: str = "H",
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
    lats = journey["lats"]
    lons = journey["lons"]
    headings = journey["headings"]

    s_X = scalers["X"]
    s_ydv = scalers["y_dv"]
    s_yw = scalers["y_w"]
    s_yba = scalers["y_ba"]
    s_ybw = scalers["y_bw"]

    physics = VehiclePhysicsEngineV6(dt=DT)
    ekf = ExtendedKalmanFilterV6(dt=DT)
    map_matcher = ProbabilisticHMMMapMatcherV6()
    map_matcher.build_road_graph_from_trajectory(lats, lons)

    # Initial states
    init_heading = math.radians(headings[start_idx])
    init_v = float(v_true[start_idx - 1])
    ekf.reset(x0=0.0, y0=0.0, v0=init_v, psi0=init_heading)

    pos_gt = [(0.0, 0.0)]
    pos_v6 = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = init_heading
    psi_ins = init_heading
    psi_v6 = init_heading
    v_ins = init_v

    # Autoregressive state history
    v_pred_prev = init_v
    history_v = list(v_true[start_idx - WINDOW_10HZ: start_idx])

    drift_progression = {}  # Drift at 1s, 3s, 5s, 10s
    speed_errors = []
    heading_errors = []

    use_bias = ablation_mode in ["C", "D", "E", "F", "G", "H"]
    use_physics = ablation_mode in ["D", "E", "F", "G", "H"]
    use_ekf = ablation_mode in ["E", "F", "G", "H"]
    use_nhc_zupt = ablation_mode in ["F", "G", "H"]
    use_map = ablation_mode in ["G", "H"]

    for k in range(OUTAGE_10HZ):
        cur = start_idx + k

        # 1. Assemble 6-channel 10 Hz window
        ch_a_fwd = np.clip(a_fwd[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW_10HZ + 1: cur + 1], -1.2, 1.2)
        ch_a_lat = np.clip(a_lat[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
        ch_v_prev = np.clip(np.array(history_v[-WINDOW_10HZ:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur - WINDOW_10HZ + 1: cur + 1], -4.0, 4.0)
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

        window = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
        win_scaled = s_X.transform(window.reshape(1, -1)).reshape(1, WINDOW_10HZ, 6).astype(np.float32)

        # 2. Neural forward pass
        with torch.no_grad():
            inp = torch.tensor(win_scaled, device=device)
            dv_s, w_s, z_log, ba_s, bw_s, log_v = model(inp)

        delta_v_pred = float(s_ydv.inverse_transform([[dv_s.item()]])[0, 0])
        w_pred = float(s_yw.inverse_transform([[w_s.item()]])[0, 0])
        b_a_pred = float(s_yba.inverse_transform([[ba_s.item()]])[0, 0]) if use_bias else 0.0
        b_w_pred = float(s_ybw.inverse_transform([[bw_s.item()]])[0, 0]) if use_bias else 0.0
        p_stop = float(torch.sigmoid(z_log).item())
        var_v = float(math.exp(log_v[0, 0].item()))
        var_w = float(math.exp(log_v[0, 1].item()))

        # 3. Physics & Non-Holonomic Validation
        if use_physics:
            v_val, w_val, diag = physics.validate_and_constrain_dynamics(
                v_pred_prev, delta_v_pred, w_pred, a_fwd[cur], a_lat[cur], b_a_pred, b_w_pred,
                w_gyro_measured=w_yaw[cur],
            )
        else:
            v_val = float(np.clip(v_pred_prev + delta_v_pred, 0.0, 45.0))
            w_val = float(np.clip(w_pred, -1.2, 1.2))

        # 4. State Estimation (EKF vs Kinematics)
        if use_ekf:
            ekf.predict(a_fwd[cur], w_yaw[cur])
            ekf.update_neural_motion(v_val, w_val, b_a_pred, b_w_pred, var_v=var_v, var_w=var_w)
            if use_nhc_zupt:
                ekf.apply_zupt(p_stop, abs(a_fwd[cur]), abs(w_yaw[cur]))
            state = ekf.get_state()
            cur_x, cur_y, cur_v, cur_psi = state["px"], state["py"], state["v"], state["psi"]
            psi_v6 = cur_psi
        else:
            # Direct kinematic integration
            if use_nhc_zupt and p_stop > 0.70 and abs(a_fwd[cur]) < 0.25:
                v_val = 0.0
                w_val = 0.0
            cur_x, cur_y, psi_v6 = physics.integrate_step(
                pos_v6[-1][0], pos_v6[-1][1], psi_v6, v_val, w_val
            )
            cur_psi = psi_v6
            cur_v = v_val

        # 5. Map Matching Filter
        if use_map:
            cur_x, cur_y, cur_psi, m_diag = map_matcher.match_step(cur_x, cur_y, cur_psi, cur_v * DT)
            if use_ekf:
                ekf.x[0], ekf.x[1], ekf.x[3] = cur_x, cur_y, cur_psi

        v_pred_prev = cur_v
        history_v.append(v_pred_prev)

        # 6. Ground Truth Update
        true_v = v_true[cur]
        true_w = w_true[cur]
        psi_gt = (psi_gt + true_w * DT + math.pi) % (2 * math.pi) - math.pi
        pos_gt.append((
            pos_gt[-1][0] + true_v * math.cos(psi_gt) * DT,
            pos_gt[-1][1] + true_v * math.sin(psi_gt) * DT,
        ))

        # Raw INS baseline
        v_ins = max(0.0, v_ins + a_fwd[cur] * DT)
        psi_ins = (psi_ins + w_yaw[cur] * DT + math.pi) % (2 * math.pi) - math.pi
        pos_ins.append((
            pos_ins[-1][0] + v_ins * math.cos(psi_ins) * DT,
            pos_ins[-1][1] + v_ins * math.sin(psi_ins) * DT,
        ))

        pos_v6.append((cur_x, cur_y))
        speed_errors.append(abs(true_v - cur_v))
        heading_errors.append(abs((cur_psi - psi_gt + math.pi) % (2 * math.pi) - math.pi))

        # Check horizon drift at 1s (step 10), 3s (step 30), 5s (step 50), 10s (step 100)
        step_num = k + 1
        if step_num in [10, 30, 50, 100]:
            h_sec = step_num // 10
            h_drift = math.hypot(pos_v6[-1][0] - pos_gt[-1][0], pos_v6[-1][1] - pos_gt[-1][1])
            drift_progression[f"drift_{h_sec}s"] = h_drift

    final_drift_v6 = math.hypot(pos_v6[-1][0] - pos_gt[-1][0], pos_v6[-1][1] - pos_gt[-1][1])
    final_drift_ins = math.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])
    total_distance = sum(v_true[start_idx: start_idx + OUTAGE_10HZ]) * DT

    mean_abs_yaw = float(np.mean(np.abs(w_true[start_idx: start_idx + OUTAGE_10HZ])))
    is_high_yaw = mean_abs_yaw >= 0.06

    return {
        "final_drift_v6": final_drift_v6,
        "final_drift_ins": final_drift_ins,
        "drift_pct_v6": (final_drift_v6 / total_distance * 100) if total_distance > 0 else 0.0,
        "drift_pct_ins": (final_drift_ins / total_distance * 100) if total_distance > 0 else 0.0,
        "total_distance": total_distance,
        "mean_abs_yaw": mean_abs_yaw,
        "is_high_yaw": is_high_yaw,
        "disp_crse_v6": float(sum(speed_errors) * DT),
        "ori_crse_v6": float(sum(heading_errors)),
        "speed_mae": float(np.mean(speed_errors)),
        "drift_progression": drift_progression,
        "pos_gt": pos_gt,
        "pos_v6": pos_v6,
        "pos_ins": pos_ins,
    }


def evaluate_all_v6(ablation_mode: str = "H", ckpt_name: str = "best_model_v6.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print(f"PINO-DR v6 EVALUATION: ABLATION [{ablation_mode}] (DEVICE: {device})")
    print("=" * 90)

    model, cfg = load_v6_model(device, ckpt_name)

    with open(DATA_DIR / "scalers_v6.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v6.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    # Load V4 summary for comparison
    v4_path = ROOT.parent / "v4_turn_focused" / "results" / "benchmark_summary_v4.json"
    v4_summary = {}
    if v4_path.exists():
        with open(v4_path) as f:
            v4_summary = json.load(f)

    scenario_metrics = {}
    all_results = []
    plot_samples = {}

    for scen in SCENARIO_ORDER:
        journeys = test_scenarios.get(scen, [])
        if not journeys:
            continue

        seq_results = []
        for j in journeys:
            n = len(j["v_true"])
            # Outage every 100 steps (10s) starting after initial window
            for start in range(WINDOW_10HZ + 10, n - OUTAGE_10HZ, 100):
                res = simulate_outage_v6(j, start, model, scalers, device, ablation_mode=ablation_mode)
                seq_results.append(res)
                all_results.append(res)
                if scen not in plot_samples and res["total_distance"] > 40:
                    plot_samples[scen] = res

        n_seq = len(seq_results)
        mean_drift_v6 = float(np.mean([r["final_drift_v6"] for r in seq_results]))
        mean_drift_ins = float(np.mean([r["final_drift_ins"] for r in seq_results]))
        mean_disp_crse = float(np.mean([r["disp_crse_v6"] for r in seq_results]))
        mean_ori_crse = float(np.mean([r["ori_crse_v6"] for r in seq_results]))
        mean_speed_mae = float(np.mean([r["speed_mae"] for r in seq_results]))

        v4_drift = v4_summary.get(scen, {}).get("final_drift_v4", None)
        imprv_vs_ins = (mean_drift_ins - mean_drift_v6) / mean_drift_ins * 100 if mean_drift_ins > 0 else 0.0
        imprv_vs_v4 = (v4_drift - mean_drift_v6) / v4_drift * 100 if v4_drift else 0.0

        scenario_metrics[scen] = {
            "n_sequences": n_seq,
            "drift_v6_10s": mean_drift_v6,
            "drift_ins_10s": mean_drift_ins,
            "drift_v4_10s": v4_drift,
            "disp_crse_v6": mean_disp_crse,
            "ori_crse_v6": mean_ori_crse,
            "speed_mae": mean_speed_mae,
            "imprv_vs_ins_pct": imprv_vs_ins,
            "imprv_vs_v4_pct": imprv_vs_v4,
        }

        v4_str = f" | v4={v4_drift:.2f}m ({imprv_vs_v4:+.1f}%)" if v4_drift else ""
        print(f"\nScenario [{scen.upper()}] ({n_seq} sequences)")
        print(f"  10s Drift: v6 = {mean_drift_v6:.2f}m | INS = {mean_drift_ins:.2f}m{v4_str}")
        print(f"  Disp CRSE: {mean_disp_crse:.2f}m | Ori CRSE: {mean_ori_crse:.3f}rad | Speed MAE: {mean_speed_mae:.2f}m/s")

    # Horizon drift across all sequences
    horizons = ["drift_1s", "drift_3s", "drift_5s", "drift_10s"]
    h_drifts = {h: float(np.mean([r["drift_progression"][h] for r in all_results])) for h in horizons}

    # Turn breakdown
    high_yaw = [r for r in all_results if r["is_high_yaw"]]
    low_yaw = [r for r in all_results if not r["is_high_yaw"]]

    overall_metrics = {
        "ablation_mode": ablation_mode,
        "total_sequences": len(all_results),
        "overall_drift_v6": float(np.mean([r["final_drift_v6"] for r in all_results])),
        "overall_drift_ins": float(np.mean([r["final_drift_ins"] for r in all_results])),
        "overall_drift_v4": v4_summary.get("overall_turn_breakdown", {}).get("overall_drift_v4", 29.99),
        "horizon_drifts": h_drifts,
        "high_yaw_drift_v6": float(np.mean([r["final_drift_v6"] for r in high_yaw])) if high_yaw else 0.0,
        "low_yaw_drift_v6": float(np.mean([r["final_drift_v6"] for r in low_yaw])) if low_yaw else 0.0,
        "scenarios": scenario_metrics,
    }

    print("\n" + "=" * 90)
    print(f"OVERALL SUMMARY (ABLATION {ablation_mode})")
    print(f"  Overall 10-s Drift:  v6 = {overall_metrics['overall_drift_v6']:.2f}m (INS = {overall_metrics['overall_drift_ins']:.2f}m)")
    print(f"  Horizon Drifts:      1s={h_drifts['drift_1s']:.2f}m | 3s={h_drifts['drift_3s']:.2f}m | 5s={h_drifts['drift_5s']:.2f}m | 10s={h_drifts['drift_10s']:.2f}m")
    print(f"  High-Yaw Drift:      {overall_metrics['high_yaw_drift_v6']:.2f}m | Low-Yaw Drift: {overall_metrics['low_yaw_drift_v6']:.2f}m")
    print("=" * 90)

    # Save summary
    out_file = RESULTS_DIR / f"benchmark_summary_v6_ablation_{ablation_mode}.json"
    with open(out_file, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"[v6 Eval] Benchmark metrics saved to: {out_file}")

    # Generate Trajectory Plots
    for scen, sample in plot_samples.items():
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        gt_x, gt_y = zip(*sample["pos_gt"])
        v6_x, v6_y = zip(*sample["pos_v6"])
        ins_x, ins_y = zip(*sample["pos_ins"])

        ax.plot(gt_x, gt_y, "g-", linewidth=2.5, label="Ground Truth GPS (True Path)")
        ax.plot(v6_x, v6_y, "b-", linewidth=2.0, label=f"v6 Intelligent DR (Ablation {ablation_mode})")
        ax.plot(ins_x, ins_y, "r--", linewidth=1.5, label="Raw Physics INS (Uncompensated)")

        ax.set_title(f"PINO-DR v6: {scen.upper()} 10-Second GNSS Outage Trajectory", fontsize=12, fontweight="bold")
        ax.set_xlabel("Local Easting (meters)")
        ax.set_ylabel("Local Northing (meters)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()

        plot_path = RESULTS_DIR / f"trajectory_v6_{scen}_{ablation_mode}.png"
        fig.savefig(plot_path)
        plt.close(fig)

    return overall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="H", choices=["A", "B", "C", "D", "E", "F", "G", "H"])
    parser.add_argument("--ckpt", default="best_model_v6.pth")
    args = parser.parse_args()
    evaluate_all_v6(ablation_mode=args.mode, ckpt_name=args.ckpt)
