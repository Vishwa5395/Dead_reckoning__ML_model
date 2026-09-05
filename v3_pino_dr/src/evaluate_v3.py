"""
evaluate_v3.py
--------------
Closed-loop autoregressive benchmark for PINO-DR v3.

Simulates 10-second GNSS outage sequences across the 5 official IO-VNBD
test scenarios, with the ZUPT hysteresis gate active.
Generates comparison metrics (v2 vs v3 vs Raw INS) and trajectory plots.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models_v3 import PINODeadReckoningNet

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "preprocessed" / "v3"
CKPT_DIR = ROOT / "checkpoints_v3"
RESULTS_DIR = ROOT / "results"

# Physical bounds (must match preprocess_v3 CLIP)
DISP_LO, DISP_HI = 0.0, 45.0
ORI_LO, ORI_HI = -1.2, 1.2

WINDOW = 10
OUTAGE = 10

SCENARIO_ORDER = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]


# ─── ZUPT hysteresis gate ────────────────────────────────────────────────────

class ZUPTHysteresisGate:
    """
    Hysteresis-based ZUPT gate to prevent stutter at the decision boundary.
    - Enter stopped: N_enter consecutive steps with p_stop > threshold_high.
    - Exit stopped:  N_exit  consecutive steps with p_stop < threshold_low.
    """

    def __init__(self, threshold_high=0.7, threshold_low=0.3,
                 n_enter=3, n_exit=2):
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


# ─── Model loading ───────────────────────────────────────────────────────────

def load_v3_model(device: torch.device):
    ckpt_path = CKPT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best model not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PINODeadReckoningNet(
        in_channels=cfg.get("in_channels", 4),
        conv_channels=cfg.get("conv_channels", 32),
        gru_hidden=cfg.get("gru_hidden", 32),
        num_gru_layers=cfg.get("num_gru_layers", 1),
        dropout=cfg.get("dropout", 0.20),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─── Closed-loop simulation ─────────────────────────────────────────────────

def simulate_sequence_v3(journey, start_idx, model, scalers, device):
    """Simulate a 10-second GNSS outage using the v3 PINO-DR model."""
    a_fwd = journey["a_fwd"]
    w_yaw = journey["w_yaw"]
    a_lat = journey["a_lat"]
    x_gps = journey["x_gps"]
    w_gps = journey["w_gps"]
    headings = journey["headings"]

    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]

    # History for autoregressive v_prev feedback
    history_x_pred = list(x_gps[start_idx - WINDOW: start_idx])

    pos_gt = [(0.0, 0.0)]
    pos_v3 = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = np.radians(headings[start_idx])
    psi_v3 = psi_gt
    psi_ins = psi_gt
    v_ins = x_gps[start_idx - 1]

    disp_err, ins_err = [], []
    ori_err_v3, ori_err_ins = [], []

    zupt_gate = ZUPTHysteresisGate()

    for k in range(OUTAGE):
        cur = start_idx + k

        # Build (1, 10, 4) input window
        ch_a_fwd = np.clip(a_fwd[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW + 1: cur + 1], *(-1.0, 1.0))
        ch_a_lat = np.clip(a_lat[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_v_prev = np.clip(np.array(history_x_pred[-WINDOW:]), 0.0, 45.0)

        window = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev], axis=-1)  # (10, 4)
        window_flat = window.reshape(1, -1)  # (1, 40)
        window_scaled = s_X.transform(window_flat).reshape(1, 10, 4).astype(np.float32)

        with torch.no_grad():
            inp = torch.tensor(window_scaled, dtype=torch.float32).to(device)
            d_pred_s, o_pred_s, z_logit = model(inp)

        # Inverse-transform predictions
        x_pred_raw = s_yd.inverse_transform([[d_pred_s.item()]])[0, 0]
        w_pred_raw = s_yo.inverse_transform([[o_pred_s.item()]])[0, 0]

        x_pred = float(np.clip(x_pred_raw, DISP_LO, DISP_HI))
        w_pred = float(np.clip(w_pred_raw, ORI_LO, ORI_HI))
        p_stop = float(torch.sigmoid(z_logit).item())

        # ZUPT hysteresis gate
        is_stopped = zupt_gate.update(p_stop)
        if is_stopped:
            x_pred = 0.0
            w_pred = 0.0

        history_x_pred.append(x_pred)

        # Ground truth
        true_x = x_gps[cur]
        true_w = w_gps[cur]
        raw_w = w_yaw[cur]

        # Heading updates
        psi_gt += true_w
        psi_v3 += w_pred
        psi_ins += raw_w

        # INS baseline
        v_ins = max(0.0, v_ins + a_fwd[cur])
        x_ins = v_ins

        # Errors
        disp_err.append(abs(true_x - x_pred))
        ins_err.append(abs(true_x - x_ins))
        ori_err_v3.append(abs(true_w - w_pred))
        ori_err_ins.append(abs(true_w - raw_w))

        # Position updates
        pos_gt.append((pos_gt[-1][0] + true_x * np.cos(psi_gt),
                       pos_gt[-1][1] + true_x * np.sin(psi_gt)))
        pos_v3.append((pos_v3[-1][0] + x_pred * np.cos(psi_v3),
                       pos_v3[-1][1] + x_pred * np.sin(psi_v3)))
        pos_ins.append((pos_ins[-1][0] + x_ins * np.cos(psi_ins),
                        pos_ins[-1][1] + x_ins * np.sin(psi_ins)))

    total_dist = sum(x_gps[start_idx: start_idx + OUTAGE])
    drift_v3 = np.hypot(pos_v3[-1][0] - pos_gt[-1][0], pos_v3[-1][1] - pos_gt[-1][1])
    drift_ins = np.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])

    return {
        "total_distance": total_dist,
        "disp_crse_v3": sum(disp_err),
        "disp_crse_ins": sum(ins_err),
        "disp_aeps_v3": float(np.mean(disp_err)),
        "disp_aeps_ins": float(np.mean(ins_err)),
        "ori_crse_v3": sum(ori_err_v3),
        "ori_crse_ins": sum(ori_err_ins),
        "ori_aeps_v3": float(np.mean(ori_err_v3)),
        "ori_aeps_ins": float(np.mean(ori_err_ins)),
        "final_drift_v3": float(drift_v3),
        "final_drift_ins": float(drift_ins),
        "drift_pct_v3": (drift_v3 / total_dist * 100) if total_dist > 0 else 0.0,
        "drift_pct_ins": (drift_ins / total_dist * 100) if total_dist > 0 else 0.0,
        "pos_gt": np.array(pos_gt),
        "pos_v3": np.array(pos_v3),
        "pos_ins": np.array(pos_ins),
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_trajectory_v3(res, scen_name, results_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    gt = res["pos_gt"]
    v3 = res["pos_v3"]
    ins = res["pos_ins"]
    ax.plot(gt[:, 1], gt[:, 0], "g-", linewidth=2.5, label="Ground Truth GPS")
    ax.plot(v3[:, 1], v3[:, 0], "b--", linewidth=2.0, label="PINO-DR v3 (Closed-Loop)")
    ax.plot(ins[:, 1], ins[:, 0], "r:", linewidth=1.8, label="Pure INS Dead Reckoning")
    ax.scatter([gt[0, 1]], [gt[0, 0]], color="black", marker="o", s=80, zorder=5, label="Outage Start")
    ax.scatter([gt[-1, 1]], [gt[-1, 0]], color="green", marker="X", s=90, zorder=5, label="GPS End")
    ax.scatter([v3[-1, 1]], [v3[-1, 0]], color="blue", marker="^", s=90, zorder=5, label="v3 End")
    ax.set_title(
        f"10-s GNSS Outage: {scen_name.replace('_', ' ').title()}\n"
        f"v3 Drift: {res['final_drift_v3']:.1f}m ({res['drift_pct_v3']:.1f}%) vs "
        f"INS: {res['final_drift_ins']:.1f}m ({res['drift_pct_ins']:.1f}%)"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    path = results_dir / f"trajectory_v3_{scen_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ─── Full evaluation ─────────────────────────────────────────────────────────

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CACHE_DIR / "scalers_v3.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(CACHE_DIR / "test_scenarios_v3.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    model = load_v3_model(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load v2 results for comparison if available
    v2_summary = {}
    v2_path = RESULTS_DIR / "benchmark_summary_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2_summary = json.load(f)

    summary = {}
    print("\n" + "=" * 80)
    print("PINO-DR v3 BENCHMARK (AUTOREGRESSIVE CLOSED-LOOP, ZUPT HYSTERESIS)")
    print("=" * 80)

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
                res = simulate_sequence_v3(j, start, model, scalers, device)
                seq_results.append(res)
                if sample is None and res["total_distance"] > 50:
                    sample = res

        n_seq = len(seq_results)
        keys = [
            "disp_crse_v3", "disp_crse_ins", "disp_aeps_v3", "disp_aeps_ins",
            "ori_crse_v3", "ori_crse_ins", "ori_aeps_v3", "ori_aeps_ins",
            "final_drift_v3", "final_drift_ins", "drift_pct_v3", "drift_pct_ins",
        ]
        rec = {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
        rec["n_sequences"] = n_seq
        rec["disp_improvement_vs_ins_pct"] = (
            (rec["disp_crse_ins"] - rec["disp_crse_v3"]) / rec["disp_crse_ins"] * 100
            if rec["disp_crse_ins"] > 0 else 0.0
        )
        rec["ori_improvement_vs_ins_pct"] = (
            (rec["ori_crse_ins"] - rec["ori_crse_v3"]) / rec["ori_crse_ins"] * 100
            if rec["ori_crse_ins"] > 0 else 0.0
        )

        # v2 comparison
        if scen_name in v2_summary:
            v2 = v2_summary[scen_name]
            rec["v2_disp_crse"] = v2.get("disp_crse_idnn", None)
            rec["v2_ori_crse"] = v2.get("ori_crse_idnn", None)
            rec["v2_drift"] = v2.get("final_drift_idnn", None)

        summary[scen_name] = rec

        print(f"\nScenario [{scen_name.upper()}] ({n_seq} sequences)")
        print(f"  Disp CRSE: v3={rec['disp_crse_v3']:.2f}m  INS={rec['disp_crse_ins']:.2f}m  "
              f"(Imprv {rec['disp_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  Ori  CRSE: v3={rec['ori_crse_v3']:.3f}rad  INS={rec['ori_crse_ins']:.3f}rad  "
              f"(Imprv {rec['ori_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  10s Drift: v3={rec['final_drift_v3']:.2f}m ({rec['drift_pct_v3']:.1f}%)  "
              f"INS={rec['final_drift_ins']:.2f}m ({rec['drift_pct_ins']:.1f}%)")
        if rec.get("v2_drift") is not None:
            print(f"  vs v2:     v2_drift={rec['v2_drift']:.2f}m  "
                  f"v3_drift={rec['final_drift_v3']:.2f}m")

        if sample is not None:
            p = plot_trajectory_v3(sample, scen_name, RESULTS_DIR)
            print(f"  [Plot] {p}")

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

    with open(RESULTS_DIR / "benchmark_summary_v3.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    with open(RESULTS_DIR / "benchmark_summary_v3.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\n[v3] Benchmark summary saved: {RESULTS_DIR / 'benchmark_summary_v3.json'}")

    # ── Latency benchmark ──
    print("\n[v3] Running latency benchmark (1000 iterations)...")
    model.eval()
    dummy = torch.randn(1, 10, 4, device=device)
    # Warm-up
    for _ in range(50):
        with torch.no_grad():
            model(dummy)

    import time
    t0 = time.perf_counter()
    n_iter = 1000
    for _ in range(n_iter):
        with torch.no_grad():
            model(dummy)
    t_total = (time.perf_counter() - t0) * 1000  # ms
    avg_ms = t_total / n_iter
    print(f"[v3] Latency: {avg_ms:.3f} ms/step ({n_iter} iterations on {device})")

    return summary


if __name__ == "__main__":
    evaluate_all()
