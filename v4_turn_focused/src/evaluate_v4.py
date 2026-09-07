"""
evaluate_v4.py
--------------
Closed-loop autoregressive benchmark for PINO-DR v4 (Turn-Focused).

Evaluates 10-second GNSS outage sequences across all 5 standard IO-VNBD scenarios
with ZUPT hysteresis gating active and strictly model-derived autoregressive state feedback.

Reports:
1. Overall 10-s drift (m)
2. High-yaw vs Low-yaw drift (m)
3. High-yaw vs Low-yaw orientation CRSE (rad)
4. Comprehensive 5-scenario comparison: Raw INS vs v1 vs v2 vs v3 vs v4
5. Trajectory plots for all scenarios
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models_v4 import PINODeadReckoningNetV4

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


# ─── ZUPT Hysteresis Gate ───────────────────────────────────────────────────

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


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_v4_model(device: torch.device, ckpt_name: str = "best_model.pth"):
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = PINODeadReckoningNetV4(
        in_channels=cfg.get("in_channels", 6),
        conv_channels=cfg.get("conv_channels", 32),
        gru_hidden=cfg.get("gru_hidden", 32),
        num_gru_layers=cfg.get("num_gru_layers", 1),
        dropout=cfg.get("dropout", 0.20),
        use_multihead_attention=cfg.get("use_multihead_attention", True),
        use_cross_task_coupling=cfg.get("use_cross_task_coupling", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ─── Autoregressive Sequence Simulation ──────────────────────────────────────

def simulate_sequence_v4(journey: dict, start_idx: int, model: PINODeadReckoningNetV4,
                         cfg: dict, scalers: dict, device: torch.device):
    """
    Simulates a 10-second GNSS outage using strictly closed-loop predictions.
    centripetal_residual is calculated strictly using predicted v_prev.
    """
    a_fwd = journey["a_fwd"]
    w_yaw = journey["w_yaw"]
    a_lat = journey["a_lat"]
    w_accel = journey["w_yaw_accel"]
    x_gps = journey["x_gps"]
    w_gps = journey["w_gps"]
    headings = journey["headings"]

    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]
    in_channels = cfg.get("in_channels", 6)

    history_x_pred = list(x_gps[start_idx - WINDOW: start_idx])

    pos_gt = [(0.0, 0.0)]
    pos_v4 = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = np.radians(headings[start_idx])
    psi_v4 = psi_gt
    psi_ins = psi_gt
    v_ins = x_gps[start_idx - 1]

    disp_err, ins_err = [], []
    ori_err_v4, ori_err_ins = [], []

    zupt_gate = ZUPTHysteresisGate()

    for k in range(OUTAGE):
        cur = start_idx + k

        # 6-channel input window assembly
        ch_a_fwd = np.clip(a_fwd[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW + 1: cur + 1], *(-1.0, 1.0))
        ch_a_lat = np.clip(a_lat[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_v_prev = np.clip(np.array(history_x_pred[-WINDOW:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur - WINDOW + 1: cur + 1], *(-2.0, 2.0))

        # Centripetal residual strictly from predicted v_prev
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, *(-8.0, 8.0))

        all_channels = [ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal]
        window = np.stack(all_channels[:in_channels], axis=-1)  # (10, in_channels)

        # Scale window
        if in_channels == 6:
            window_flat = window.reshape(1, -1)
            window_scaled = s_X.transform(window_flat).reshape(1, 10, in_channels).astype(np.float32)
        else:
            # For ablation models with 4 or 5 channels
            dummy_full = np.zeros((1, 10, 6), dtype=np.float32)
            dummy_full[0, :, :in_channels] = window
            dummy_scaled = s_X.transform(dummy_full.reshape(1, -1)).reshape(1, 10, 6)
            window_scaled = dummy_scaled[:, :, :in_channels].astype(np.float32)

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
        psi_v4 += w_pred
        psi_ins += raw_w

        v_ins = max(0.0, v_ins + a_fwd[cur])
        x_ins = v_ins

        disp_err.append(abs(true_x - x_pred))
        ins_err.append(abs(true_x - x_ins))
        ori_err_v4.append(abs(true_w - w_pred))
        ori_err_ins.append(abs(true_w - raw_w))

        pos_gt.append((pos_gt[-1][0] + true_x * np.cos(psi_gt),
                       pos_gt[-1][1] + true_x * np.sin(psi_gt)))
        pos_v4.append((pos_v4[-1][0] + x_pred * np.cos(psi_v4),
                       pos_v4[-1][1] + x_pred * np.sin(psi_v4)))
        pos_ins.append((pos_ins[-1][0] + x_ins * np.cos(psi_ins),
                        pos_ins[-1][1] + x_ins * np.sin(psi_ins)))

    total_dist = sum(x_gps[start_idx: start_idx + OUTAGE])
    drift_v4 = np.hypot(pos_v4[-1][0] - pos_gt[-1][0], pos_v4[-1][1] - pos_gt[-1][1])
    drift_ins = np.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])

    mean_abs_yaw = float(np.mean(np.abs(w_gps[start_idx: start_idx + OUTAGE])))
    is_high_yaw = mean_abs_yaw >= HIGH_YAW_THRESHOLD

    return {
        "total_distance": total_dist,
        "mean_abs_yaw": mean_abs_yaw,
        "is_high_yaw": is_high_yaw,
        "disp_crse_v4": sum(disp_err),
        "disp_crse_ins": sum(ins_err),
        "disp_aeps_v4": float(np.mean(disp_err)),
        "disp_aeps_ins": float(np.mean(ins_err)),
        "ori_crse_v4": sum(ori_err_v4),
        "ori_crse_ins": sum(ori_err_ins),
        "ori_aeps_v4": float(np.mean(ori_err_v4)),
        "ori_aeps_ins": float(np.mean(ori_err_ins)),
        "final_drift_v4": float(drift_v4),
        "final_drift_ins": float(drift_ins),
        "drift_pct_v4": (drift_v4 / total_dist * 100) if total_dist > 0 else 0.0,
        "drift_pct_ins": (drift_ins / total_dist * 100) if total_dist > 0 else 0.0,
        "pos_gt": np.array(pos_gt),
        "pos_v4": np.array(pos_v4),
        "pos_ins": np.array(pos_ins),
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_trajectory_v4(res: dict, scen_name: str, results_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    gt = res["pos_gt"]
    v4 = res["pos_v4"]
    ins = res["pos_ins"]

    ax.plot(gt[:, 1], gt[:, 0], "g-", linewidth=2.5, label="Ground Truth GPS")
    ax.plot(v4[:, 1], v4[:, 0], "b--", linewidth=2.0, label="PINO-DR v4 (Turn-Focused)")
    ax.plot(ins[:, 1], ins[:, 0], "r:", linewidth=1.8, label="Pure INS Dead Reckoning")

    ax.scatter([gt[0, 1]], [gt[0, 0]], color="black", marker="o", s=80, zorder=5, label="Outage Start")
    ax.scatter([gt[-1, 1]], [gt[-1, 0]], color="green", marker="X", s=90, zorder=5, label="GPS End")
    ax.scatter([v4[-1, 1]], [v4[-1, 0]], color="blue", marker="^", s=90, zorder=5, label="v4 End")

    ax.set_title(
        f"10-s GNSS Outage: {scen_name.replace('_', ' ').title()}\n"
        f"v4 Drift: {res['final_drift_v4']:.1f}m ({res['drift_pct_v4']:.1f}%) vs "
        f"INS: {res['final_drift_ins']:.1f}m ({res['drift_pct_ins']:.1f}%)"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    path = results_dir / f"trajectory_v4_{scen_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ─── Full Benchmark ──────────────────────────────────────────────────────────

def evaluate_all(ckpt_name: str = "best_model.pth", out_suffix: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATA_DIR / "scalers_v4.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v4.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    model, cfg = load_v4_model(device, ckpt_name=ckpt_name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load v3 summary for comparison
    v3_summary = {}
    v3_path = ROOT.parent / "v3_pino_dr" / "results" / "benchmark_summary_v3.json"
    if v3_path.exists():
        with open(v3_path) as f:
            v3_summary = json.load(f)

    summary = {}
    all_seq_results = []

    print("\n" + "=" * 90)
    print("PINO-DR v4 BENCHMARK (CLOSED-LOOP AUTOREGRESSIVE, TURN DYNAMICS REFINEMENT)")
    print("=" * 90)

    for scen_name in SCENARIO_ORDER:
        journeys = test_scenarios.get(scen_name, [])
        if not journeys:
            print(f"[skip] {scen_name}: no journeys")
            continue

        seq_results = []
        sample = None
        for j in journeys:
            n = len(j["x_gps"])
            for start in range(WINDOW + 1, n - OUTAGE, 10):
                res = simulate_sequence_v4(j, start, model, cfg, scalers, device)
                seq_results.append(res)
                all_seq_results.append(res)
                if sample is None and res["total_distance"] > 50:
                    sample = res

        n_seq = len(seq_results)
        keys = [
            "disp_crse_v4", "disp_crse_ins", "disp_aeps_v4", "disp_aeps_ins",
            "ori_crse_v4", "ori_crse_ins", "ori_aeps_v4", "ori_aeps_ins",
            "final_drift_v4", "final_drift_ins", "drift_pct_v4", "drift_pct_ins",
        ]
        rec = {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
        rec["n_sequences"] = n_seq
        rec["disp_improvement_vs_ins_pct"] = (
            (rec["disp_crse_ins"] - rec["disp_crse_v4"]) / rec["disp_crse_ins"] * 100
            if rec["disp_crse_ins"] > 0 else 0.0
        )
        rec["ori_improvement_vs_ins_pct"] = (
            (rec["ori_crse_ins"] - rec["ori_crse_v4"]) / rec["ori_crse_ins"] * 100
            if rec["ori_crse_ins"] > 0 else 0.0
        )

        # High-yaw vs Low-yaw breakdown within this scenario
        high_yaw_seqs = [r for r in seq_results if r["is_high_yaw"]]
        low_yaw_seqs = [r for r in seq_results if not r["is_high_yaw"]]

        rec["high_yaw_drift"] = float(np.mean([r["final_drift_v4"] for r in high_yaw_seqs])) if high_yaw_seqs else None
        rec["low_yaw_drift"] = float(np.mean([r["final_drift_v4"] for r in low_yaw_seqs])) if low_yaw_seqs else None
        rec["high_yaw_ori_crse"] = float(np.mean([r["ori_crse_v4"] for r in high_yaw_seqs])) if high_yaw_seqs else None
        rec["low_yaw_ori_crse"] = float(np.mean([r["ori_crse_v4"] for r in low_yaw_seqs])) if low_yaw_seqs else None

        # v3 baseline comparison
        if scen_name in v3_summary:
            v3 = v3_summary[scen_name]
            rec["v3_drift"] = v3.get("final_drift_v3", None)
            rec["v3_disp_crse"] = v3.get("disp_crse_v3", None)
            rec["v3_ori_crse"] = v3.get("ori_crse_v3", None)
            if rec["v3_drift"]:
                rec["drift_imprv_vs_v3_pct"] = (rec["v3_drift"] - rec["final_drift_v4"]) / rec["v3_drift"] * 100

        summary[scen_name] = rec

        v3_str = f"  v3_drift={rec.get('v3_drift', 0):.2f}m (Imprv {rec.get('drift_imprv_vs_v3_pct', 0):+.1f}%)" if rec.get("v3_drift") else ""
        print(f"\nScenario [{scen_name.upper()}] ({n_seq} sequences)")
        print(f"  Disp CRSE: v4={rec['disp_crse_v4']:.2f}m  INS={rec['disp_crse_ins']:.2f}m  (Imprv {rec['disp_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  Ori  CRSE: v4={rec['ori_crse_v4']:.3f}rad INS={rec['ori_crse_ins']:.3f}rad (Imprv {rec['ori_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  10s Drift: v4={rec['final_drift_v4']:.2f}m ({rec['drift_pct_v4']:.1f}%)  INS={rec['final_drift_ins']:.2f}m{v3_str}")
        hy_str = f"{rec['high_yaw_drift']:.2f}m" if rec["high_yaw_drift"] is not None else "N/A"
        ly_str = f"{rec['low_yaw_drift']:.2f}m" if rec["low_yaw_drift"] is not None else "N/A"
        print(f"  High-Yaw Drift: {hy_str} | Low-Yaw Drift: {ly_str}")

        if sample is not None:
            p = plot_trajectory_v4(sample, scen_name, RESULTS_DIR)
            print(f"  [Plot] {p}")

    # ── Overall High-Yaw vs Low-Yaw Summary across all scenarios ──
    all_high_yaw = [r for r in all_seq_results if r["is_high_yaw"]]
    all_low_yaw = [r for r in all_seq_results if not r["is_high_yaw"]]

    overall_metrics = {
        "total_sequences": len(all_seq_results),
        "overall_drift_v4": float(np.mean([r["final_drift_v4"] for r in all_seq_results])),
        "overall_drift_ins": float(np.mean([r["final_drift_ins"] for r in all_seq_results])),
        "high_yaw_sequences": len(all_high_yaw),
        "high_yaw_drift_v4": float(np.mean([r["final_drift_v4"] for r in all_high_yaw])) if all_high_yaw else 0.0,
        "high_yaw_drift_ins": float(np.mean([r["final_drift_ins"] for r in all_high_yaw])) if all_high_yaw else 0.0,
        "high_yaw_ori_crse_v4": float(np.mean([r["ori_crse_v4"] for r in all_high_yaw])) if all_high_yaw else 0.0,
        "high_yaw_ori_crse_ins": float(np.mean([r["ori_crse_ins"] for r in all_high_yaw])) if all_high_yaw else 0.0,
        "low_yaw_sequences": len(all_low_yaw),
        "low_yaw_drift_v4": float(np.mean([r["final_drift_v4"] for r in all_low_yaw])) if all_low_yaw else 0.0,
        "low_yaw_drift_ins": float(np.mean([r["final_drift_ins"] for r in all_low_yaw])) if all_low_yaw else 0.0,
        "low_yaw_ori_crse_v4": float(np.mean([r["ori_crse_v4"] for r in all_low_yaw])) if all_low_yaw else 0.0,
        "low_yaw_ori_crse_ins": float(np.mean([r["ori_crse_ins"] for r in all_low_yaw])) if all_low_yaw else 0.0,
    }
    summary["overall_turn_breakdown"] = overall_metrics

    print("\n" + "=" * 90)
    print("TURN BREAKDOWN SUMMARY (ALL TEST SEQUENCES):")
    print(f"  Overall 10-s Drift:     v4 = {overall_metrics['overall_drift_v4']:.2f}m | INS = {overall_metrics['overall_drift_ins']:.2f}m")
    print(f"  High-Yaw Drift ({overall_metrics['high_yaw_sequences']} seqs): v4 = {overall_metrics['high_yaw_drift_v4']:.2f}m | INS = {overall_metrics['high_yaw_drift_ins']:.2f}m")
    print(f"  High-Yaw Ori CRSE:      v4 = {overall_metrics['high_yaw_ori_crse_v4']:.3f}rad | INS = {overall_metrics['high_yaw_ori_crse_ins']:.3f}rad")
    print(f"  Low-Yaw Drift ({overall_metrics['low_yaw_sequences']} seqs):  v4 = {overall_metrics['low_yaw_drift_v4']:.2f}m | INS = {overall_metrics['low_yaw_drift_ins']:.2f}m")
    print(f"  Low-Yaw Ori CRSE:       v4 = {overall_metrics['low_yaw_ori_crse_v4']:.3f}rad | INS = {overall_metrics['low_yaw_ori_crse_ins']:.3f}rad")
    print("=" * 90)

    # ── Save summary ──
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)

    json_name = f"benchmark_summary_v4{out_suffix}.json"
    with open(RESULTS_DIR / json_name, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    # Latency benchmark
    print("\n[v4] Running latency benchmark (1000 iterations)...")
    dummy = torch.randn(1, 10, cfg.get("in_channels", 6), device=device)
    for _ in range(50):
        with torch.no_grad():
            model(dummy)

    import time
    t0 = time.perf_counter()
    n_iter = 1000
    for _ in range(n_iter):
        with torch.no_grad():
            model(dummy)
    avg_ms = (time.perf_counter() - t0) * 1000 / n_iter
    print(f"[v4] Latency: {avg_ms:.3f} ms/step ({n_iter} iterations on {device})")
    summary["latency_ms"] = avg_ms

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PINO-DR v4 checkpoint")
    parser.add_argument("--ckpt", default="best_model.pth", help="Checkpoint filename in checkpoints/")
    parser.add_argument("--suffix", default="", help="Suffix for output json summary")
    args = parser.parse_args()
    evaluate_all(ckpt_name=args.ckpt, out_suffix=args.suffix)
