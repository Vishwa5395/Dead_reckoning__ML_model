"""
evaluate_v5.py
--------------
Closed-loop autoregressive dead-reckoning benchmark for PINO-DR v5.

Evaluates on all 65 test outage sequences across the 5 IO-VNBD scenarios:
1. motorway (7 sequences)
2. roundabout (3 sequences)
3. quick_accel (4 sequences)
4. hard_brake (12 sequences)
5. sharp_turns (39 sequences)

Outputs exact per-regime drift, CRSE, and comparison against Model D, Ablation C, and Model E.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from v5_wheel_aided.src.models_v5 import PINODeadReckoningNetV5

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DISP_LO, DISP_HI = 0.0, 45.0
ORI_LO, ORI_HI = -1.2, 1.2
WINDOW = 10
OUTAGE = 10

SCENARIO_ORDER = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
HIGH_YAW_THRESHOLD = 0.06  # rad/s mean absolute yaw rate over 10s outage

BASELINES = {
    "model_d": {
        "motorway": 10.19,
        "roundabout": 55.36,
        "quick_accel": 19.58,
        "hard_brake": 17.88,
        "sharp_turns": 36.39,
        "overall": 29.99,
    },
    "ablation_c_fail": {
        "motorway": 19.95,
        "roundabout": 77.19,
        "quick_accel": 28.04,
        "hard_brake": 18.08,
        "sharp_turns": 39.06,
        "overall": 34.21,
    },
    "model_e_fail": {
        "motorway": 12.47,
        "roundabout": 92.63,
        "quick_accel": 17.00,
        "hard_brake": 18.38,
        "sharp_turns": 47.50,
        "overall": 41.52,
    }
}


class ZUPTHysteresisGate:
    def __init__(self, threshold_high=0.7, threshold_low=0.3, n_enter=3, n_exit=2):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def reset(self):
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def update(self, p_stop: float) -> bool:
        if not self.is_stopped:
            if p_stop > self.threshold_high:
                self.high_count += 1
                self.low_count = 0
                if self.high_count >= self.n_enter:
                    self.is_stopped = True
            else:
                self.high_count = 0
        else:
            if p_stop < self.threshold_low:
                self.low_count += 1
                self.high_count = 0
                if self.low_count >= self.n_exit:
                    self.is_stopped = False
            else:
                self.low_count = 0
        return self.is_stopped


def simulate_sequence_v5(journey: dict, start_idx: int, model: PINODeadReckoningNetV5,
                         scalers: dict, cfg: dict, device: torch.device):
    a_fwd = journey['a_fwd']
    w_yaw = journey['w_yaw']
    a_lat = journey['a_lat']
    w_accel = journey['w_yaw_accel']
    v_wheel = journey['v_wheel']
    x_gps = journey['x_gps']
    w_gps = journey['w_gps']
    headings = journey['headings']

    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]
    in_channels = cfg.get("in_channels", 7)

    history_x_pred = list(x_gps[start_idx - WINDOW: start_idx])

    pos_gt = [(0.0, 0.0)]
    pos_v5 = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = np.radians(headings[start_idx])
    psi_v5 = psi_gt
    psi_ins = psi_gt
    v_ins = x_gps[start_idx - 1]

    disp_err, ins_err = [], []
    ori_err_v5, ori_err_ins = [], []

    zupt_gate = ZUPTHysteresisGate()

    for k in range(OUTAGE):
        cur = start_idx + k

        ch_a_fwd = np.clip(a_fwd[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW + 1: cur + 1], *(-1.0, 1.0))
        ch_a_lat = np.clip(a_lat[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_v_prev = np.clip(np.array(history_x_pred[-WINDOW:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur - WINDOW + 1: cur + 1], *(-2.0, 2.0))
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, *(-8.0, 8.0))
        ch_v_wheel = np.clip(v_wheel[cur - WINDOW + 1: cur + 1], *(0.0, 45.0))

        all_channels = [ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal, ch_v_wheel]
        window = np.stack(all_channels[:in_channels], axis=-1)  # (10, in_channels)

        window_flat = window.reshape(1, -1)
        window_scaled = s_X.transform(window_flat).reshape(1, 10, in_channels).astype(np.float32)

        with torch.no_grad():
            inp = torch.tensor(window_scaled, dtype=torch.float32).to(device)
            d_pred_s, o_pred_s, z_logit = model(inp)

        x_pred_raw = s_yd.inverse_transform([[d_pred_s.item()]])[0, 0]
        w_pred_raw = s_yo.inverse_transform([[o_pred_s.item()]])[0, 0]

        x_pred = float(np.clip(x_pred_raw, DISP_LO, DISP_HI))
        w_pred = float(np.clip(w_pred_raw, ORI_LO, ORI_HI))
        p_stop = float(torch.sigmoid(z_logit).item())

        is_stopped = zupt_gate.update(p_stop)
        if is_stopped:
            x_pred = 0.0
            w_pred = 0.0

        history_x_pred.append(x_pred)

        true_x = x_gps[cur]
        true_w = w_gps[cur]
        raw_w = w_yaw[cur]

        psi_gt += true_w
        psi_v5 += w_pred
        psi_ins += raw_w

        v_ins = max(0.0, v_ins + a_fwd[cur])
        x_ins = v_ins

        disp_err.append(abs(true_x - x_pred))
        ins_err.append(abs(true_x - x_ins))
        ori_err_v5.append(abs(true_w - w_pred))
        ori_err_ins.append(abs(true_w - raw_w))

        pos_gt.append((pos_gt[-1][0] + true_x * np.cos(psi_gt),
                       pos_gt[-1][1] + true_x * np.sin(psi_gt)))
        pos_v5.append((pos_v5[-1][0] + x_pred * np.cos(psi_v5),
                       pos_v5[-1][1] + x_pred * np.sin(psi_v5)))
        pos_ins.append((pos_ins[-1][0] + x_ins * np.cos(psi_ins),
                        pos_ins[-1][1] + x_ins * np.sin(psi_ins)))

    total_dist = sum(x_gps[start_idx: start_idx + OUTAGE])
    drift_v5 = np.hypot(pos_v5[-1][0] - pos_gt[-1][0], pos_v5[-1][1] - pos_gt[-1][1])
    drift_ins = np.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])

    mean_abs_yaw = float(np.mean(np.abs(w_gps[start_idx: start_idx + OUTAGE])))
    is_high_yaw = mean_abs_yaw >= HIGH_YAW_THRESHOLD

    return {
        "total_distance": total_dist,
        "mean_abs_yaw": mean_abs_yaw,
        "is_high_yaw": is_high_yaw,
        "disp_crse_v5": sum(disp_err),
        "disp_crse_ins": sum(ins_err),
        "disp_aeps_v5": float(np.mean(disp_err)),
        "disp_aeps_ins": float(np.mean(ins_err)),
        "ori_crse_v5": sum(ori_err_v5),
        "ori_crse_ins": sum(ori_err_ins),
        "ori_aeps_v5": float(np.mean(ori_err_v5)),
        "ori_aeps_ins": float(np.mean(ori_err_ins)),
        "final_drift_v5": drift_v5,
        "final_drift_ins": drift_ins,
        "drift_pct_v5": (drift_v5 / total_dist * 100) if total_dist > 0 else 0.0,
        "drift_pct_ins": (drift_ins / total_dist * 100) if total_dist > 0 else 0.0,
    }


def evaluate_all(ckpt_name: str = "best_model_v5_step1_raw_wheel.pth", out_suffix: str = "_step1"):
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(DATA_DIR / "test_scenarios_v5.pkl", "rb") as f:
        test_scenarios = pickle.load(f)
    with open(DATA_DIR / "scalers_v5.pkl", "rb") as f:
        scalers = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = PINODeadReckoningNetV5(
        in_channels=cfg["in_channels"],
        conv_channels=cfg["conv_channels"],
        gru_hidden=cfg["gru_hidden"],
        num_gru_layers=cfg["num_gru_layers"],
        dropout=cfg["dropout"],
        use_multihead_attention=cfg["use_multihead_attention"],
        use_cross_task_coupling=cfg["use_cross_task_coupling"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 90)
    print(f"PINO-DR v5 BENCHMARK EVALUATION (Checkpoint: {ckpt_name})")
    print("=" * 90)

    summary = {}
    all_seq_results = []

    for scen_name in SCENARIO_ORDER:
        journeys = test_scenarios.get(scen_name, [])
        if not journeys:
            print(f"[skip] {scen_name}: no journeys")
            continue

        seq_results = []
        for j in journeys:
            n = len(j["x_gps"])
            for start in range(WINDOW + 1, n - OUTAGE, 10):
                res = simulate_sequence_v5(j, start, model, scalers, cfg, device)
                seq_results.append(res)
                all_seq_results.append(res)

        n_seq = len(seq_results)
        keys = [
            "disp_crse_v5", "disp_crse_ins", "disp_aeps_v5", "disp_aeps_ins",
            "ori_crse_v5", "ori_crse_ins", "ori_aeps_v5", "ori_aeps_ins",
            "final_drift_v5", "final_drift_ins", "drift_pct_v5", "drift_pct_ins",
        ]
        rec = {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
        rec["n_sequences"] = n_seq
        rec["disp_improvement_vs_ins_pct"] = (
            (rec["disp_crse_ins"] - rec["disp_crse_v5"]) / rec["disp_crse_ins"] * 100
            if rec["disp_crse_ins"] > 0 else 0.0
        )
        rec["ori_improvement_vs_ins_pct"] = (
            (rec["ori_crse_ins"] - rec["ori_crse_v5"]) / rec["ori_crse_ins"] * 100
            if rec["ori_crse_ins"] > 0 else 0.0
        )

        # Baseline comparison
        mod_d_drift = BASELINES["model_d"].get(scen_name, 0.0)
        rec["model_d_drift"] = mod_d_drift
        rec["drift_diff_vs_model_d"] = rec["final_drift_v5"] - mod_d_drift
        rec["drift_imprv_vs_model_d_pct"] = (mod_d_drift - rec["final_drift_v5"]) / mod_d_drift * 100

        summary[scen_name] = rec

        print(f"\nScenario [{scen_name.upper()}] ({n_seq} sequences)")
        print(f"  Disp CRSE: v5={rec['disp_crse_v5']:.2f}m  INS={rec['disp_crse_ins']:.2f}m  (Imprv {rec['disp_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  Ori  CRSE: v5={rec['ori_crse_v5']:.3f}rad INS={rec['ori_crse_ins']:.3f}rad (Imprv {rec['ori_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  10s Drift: v5={rec['final_drift_v5']:.2f}m | Model D={mod_d_drift:.2f}m ({rec['drift_imprv_vs_model_d_pct']:+.1f}% vs Model D) | INS={rec['final_drift_ins']:.2f}m")

    # Overall Metrics
    overall_drift_v5 = float(np.mean([r["final_drift_v5"] for r in all_seq_results]))
    overall_drift_ins = float(np.mean([r["final_drift_ins"] for r in all_seq_results]))
    overall_mod_d = BASELINES["model_d"]["overall"]
    overall_imp_mod_d = (overall_mod_d - overall_drift_v5) / overall_mod_d * 100

    summary["overall"] = {
        "total_sequences": len(all_seq_results),
        "overall_drift_v5": overall_drift_v5,
        "overall_drift_ins": overall_drift_ins,
        "model_d_drift": overall_mod_d,
        "drift_diff_vs_model_d": overall_drift_v5 - overall_mod_d,
        "drift_imprv_vs_model_d_pct": overall_imp_mod_d,
    }

    print("\n" + "=" * 90)
    print("OVERALL 65-SEQUENCE DRIFT SUMMARY:")
    print(f"  v5 Drift:    {overall_drift_v5:.2f} m")
    print(f"  Model D:     {overall_mod_d:.2f} m ({overall_imp_mod_d:+.1f}% improvement vs Model D)")
    print(f"  Raw INS:     {overall_drift_ins:.2f} m ({(overall_drift_ins - overall_drift_v5)/overall_drift_ins * 100:+.1f}% vs INS)")
    print("=" * 90)

    # Save summary
    out_file = RESULTS_DIR / f"benchmark_summary_v5{out_suffix}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[v5] Saved: {out_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PINO-DR v5")
    parser.add_argument("--ckpt", default="best_model_v5_step1_raw_wheel.pth", help="Checkpoint filename in checkpoints/")
    parser.add_argument("--suffix", default="_step1", help="Suffix for output json summary")
    args = parser.parse_args()
    evaluate_all(ckpt_name=args.ckpt, out_suffix=args.suffix)
