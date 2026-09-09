"""
evaluate_v7.py
--------------
Closed-loop autoregressive benchmark evaluation for Dual-Specialist PINO-DR v7.

Key features:
1. Unified System Evaluation:
   - Evaluates the combined switching system as a whole (S1 + S2 with live yaw-rate switching).
   - Zero cheating: strictly autoregressive closed-loop execution.
2. Live Runtime Switching:
   - Computes window |yaw_rate| measure live at each step from input window.
   - Transitions smoothly between S1 (Straight Specialist) and S2 (Turning Specialist)
     with linear fading over ~0.5s at the crossover.
3. Multi-Horizon Tracking:
   - Evaluates drift progression at 1s, 3s, 5s, and 10s.
4. Comprehensive 5-Scenario IO-VNBD Coverage:
   - motorway, roundabout, quick_accel, hard_brake, sharp_turns.
5. High-Yaw vs Low-Yaw Regime Breakdowns:
   - Separate verification of straight highway tracking and sharp roundabout/cornering tracking.
6. Generates trajectory plots and exports benchmark_summary_v7.json.
"""

from __future__ import annotations

import argparse
import json
import math
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

from v7_sept_model.src.models_v7 import (
    StraightSpecialistS1,
    TurningSpecialistS2,
    DualSpecialistRuntimeSwitch,
    ProductionDualSpecialistFusion,
)

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DISP_LO, DISP_HI = 0.0, 45.0
ORI_LO, ORI_HI = -1.2, 1.2

WINDOW = 10
OUTAGE = 10

SCENARIO_ORDER = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
HIGH_YAW_BENCHMARK_THRESHOLD = 0.06  # rad/s mean absolute yaw rate over outage


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


# ─── Model Loader ────────────────────────────────────────────────────────────

def load_v7_switcher(
    device: torch.device,
    use_untuned_checkpoints: bool = False,
    threshold_yaw_rate: Optional[float] = None,
    transition_margin: float = 0.005,
    fade_time_s: float = 0.5,
    fusion_mode: str = "production",
    s1_source: str = "tuned",
):
    """
    Loads S1 and S2 models and packages them into the runtime switcher / fusion system.
    """
    if threshold_yaw_rate is None:
        meta_path = DATA_DIR / "split_metadata_v7.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold_yaw_rate = float(meta["tau_60_rad_s"])
        else:
            threshold_yaw_rate = 0.03285

    s1 = StraightSpecialistS1(in_channels=4, conv_channels=32, gru_hidden=32, num_gru_layers=1, dropout=0.20).to(device)
    s2 = TurningSpecialistS2(
        in_channels=6, conv_channels=32, gru_hidden=32, num_gru_layers=1,
        dropout=0.20, use_multihead_attention=True, use_cross_task_coupling=True,
    ).to(device)

    if use_untuned_checkpoints:
        ckpt_s1_path = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
        ckpt_s2_path = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"
        print(f"[v7][Eval] Loading UNTUNED base checkpoints:\n  S1: {ckpt_s1_path}\n  S2: {ckpt_s2_path}")
    else:
        if s1_source == "v3":
            ckpt_s1_path = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
            print(f"[v7][Eval] Using v3 Highway Anchor for S1: {ckpt_s1_path}")
        else:
            ckpt_s1_path = CKPT_DIR / "best_specialist_S1.pth"
            print(f"[v7][Eval] Using FINE-TUNED v7 weights for S1: {ckpt_s1_path}")
        ckpt_s2_path = CKPT_DIR / "best_specialist_S2.pth"
        print(f"[v7][Eval] Using FINE-TUNED v7 weights for S2: {ckpt_s2_path}")

    ckpt_s1 = torch.load(ckpt_s1_path, map_location=device, weights_only=False)
    s1.load_state_dict(ckpt_s1["model_state_dict"])
    s1.eval()

    ckpt_s2 = torch.load(ckpt_s2_path, map_location=device, weights_only=False)
    s2.load_state_dict(ckpt_s2["model_state_dict"])
    s2.eval()

    if fusion_mode == "production":
        system = ProductionDualSpecialistFusion(
            model_s1=s1,
            model_s2=s2,
            tau_enter=0.011,
            tau_exit=0.007,
            fade_time_s=fade_time_s,
        ).to(device)
        print("[v7][Eval] Activated ProductionDualSpecialistFusion (Learned Yaw Gating & Hysteresis Lock).")
    else:
        system = DualSpecialistRuntimeSwitch(
            model_s1=s1,
            model_s2=s2,
            threshold_yaw_rate=threshold_yaw_rate,
            transition_margin=transition_margin,
            fade_time_s=fade_time_s,
        ).to(device)
        print(f"[v7][Eval] Activated DualSpecialistRuntimeSwitch (Mode: {fusion_mode.upper()}).")

    system.eval()
    return system


# ─── Closed-Loop Outage Simulation ───────────────────────────────────────────

def simulate_sequence_v7(
    journey: dict,
    start_idx: int,
    switcher: DualSpecialistRuntimeSwitch,
    scalers: dict,
    device: torch.device,
    measure_mode: str = "standard",
) -> dict:
    """
    Simulates a 10-second GNSS outage using strictly closed-loop predictions
    with live yaw-rate switching between S1 and S2.
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

    history_x_pred = list(x_gps[start_idx - WINDOW: start_idx])

    pos_gt = [(0.0, 0.0)]
    pos_v7 = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = np.radians(headings[start_idx])
    psi_v7 = psi_gt
    psi_ins = psi_gt
    v_ins = x_gps[start_idx - 1]

    disp_err, ins_err = [], []
    ori_err_v7, ori_err_ins = [], []
    alphas = []

    zupt_gate = ZUPTHysteresisGate()
    switcher.reset_state()

    drift_horizons = {}

    for k in range(OUTAGE):
        cur = start_idx + k

        # Assemble full 6-channel input window
        ch_a_fwd = np.clip(a_fwd[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_w_yaw = np.clip(w_yaw[cur - WINDOW + 1: cur + 1], *(-1.0, 1.0))
        ch_a_lat = np.clip(a_lat[cur - WINDOW + 1: cur + 1], *(-8.0, 8.0))
        ch_v_prev = np.clip(np.array(history_x_pred[-WINDOW:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur - WINDOW + 1: cur + 1], *(-2.0, 2.0))

        # Centripetal residual strictly from predicted v_prev
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, *(-8.0, 8.0))

        all_channels = [ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal]
        window = np.stack(all_channels, axis=-1)  # (10, 6)

        # Scale 6-channel window
        window_flat = window.reshape(1, -1)
        window_scaled = s_X.transform(window_flat).reshape(1, 10, 6).astype(np.float32)

        if isinstance(switcher, ProductionDualSpecialistFusion):
            with torch.no_grad():
                inp = torch.tensor(window_scaled, dtype=torch.float32, device=device)
                d_pred_s, o_pred_s, z_logit, alpha, is_turning = switcher(
                    inp,
                    ch_w_yaw_phys=ch_w_yaw,
                    ch_a_lat_phys=ch_a_lat,
                    v_prev_phys=float(ch_v_prev[-1]),
                    dt=1.0,
                )
        else:
            # Live measure of |yaw_rate| across the window
            if measure_mode == "adaptive":
                # Filtered net gyro yaw rate + model prediction
                w_net = abs(float(np.mean(ch_w_yaw)))
                with torch.no_grad():
                    inp_dummy = torch.tensor(window_scaled[:, :, :4], dtype=torch.float32, device=device)
                    _, o1_prelim, _ = switcher.model_s1(inp_dummy)
                w_pred_prelim = abs(float(s_yo.inverse_transform([[o1_prelim.item()]])[0, 0]))
                measured_yaw_rate = max(w_net, w_pred_prelim)
            else:
                # Exact spec measure: mean absolute yaw rate across window
                measured_yaw_rate = float(np.mean(np.abs(ch_w_yaw)))

            with torch.no_grad():
                inp = torch.tensor(window_scaled, dtype=torch.float32, device=device)
                d_pred_s, o_pred_s, z_logit, alpha = switcher(inp, measured_yaw_rate=measured_yaw_rate, dt=1.0)

        alphas.append(alpha)

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
        psi_v7 += w_pred
        psi_ins += raw_w

        v_ins = max(0.0, v_ins + a_fwd[cur])
        x_ins = v_ins

        disp_err.append(abs(true_x - x_pred))
        ins_err.append(abs(true_x - x_ins))
        ori_err_v7.append(abs(true_w - w_pred))
        ori_err_ins.append(abs(true_w - raw_w))

        pos_gt.append((pos_gt[-1][0] + true_x * np.cos(psi_gt),
                       pos_gt[-1][1] + true_x * np.sin(psi_gt)))
        pos_v7.append((pos_v7[-1][0] + x_pred * np.cos(psi_v7),
                       pos_v7[-1][1] + x_pred * np.sin(psi_v7)))
        pos_ins.append((pos_ins[-1][0] + x_ins * np.cos(psi_ins),
                        pos_ins[-1][1] + x_ins * np.sin(psi_ins)))

        # Horizon snapshots (k=0 -> 1s, k=2 -> 3s, k=4 -> 5s, k=9 -> 10s)
        step_idx = k + 1
        if step_idx in [1, 3, 5, 10]:
            h_drift = float(np.hypot(pos_v7[-1][0] - pos_gt[-1][0], pos_v7[-1][1] - pos_gt[-1][1]))
            drift_horizons[f"drift_{step_idx}s"] = h_drift

    total_dist = sum(x_gps[start_idx: start_idx + OUTAGE])
    drift_v7 = np.hypot(pos_v7[-1][0] - pos_gt[-1][0], pos_v7[-1][1] - pos_gt[-1][1])
    drift_ins = np.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])

    mean_abs_yaw = float(np.mean(np.abs(w_gps[start_idx: start_idx + OUTAGE])))
    is_high_yaw = mean_abs_yaw >= HIGH_YAW_BENCHMARK_THRESHOLD

    return {
        "total_distance": total_dist,
        "mean_abs_yaw": mean_abs_yaw,
        "is_high_yaw": is_high_yaw,
        "mean_alpha": float(np.mean(alphas)),
        "disp_crse_v7": sum(disp_err),
        "disp_crse_ins": sum(ins_err),
        "disp_aeps_v7": float(np.mean(disp_err)),
        "disp_aeps_ins": float(np.mean(ins_err)),
        "ori_crse_v7": sum(ori_err_v7),
        "ori_crse_ins": sum(ori_err_ins),
        "ori_aeps_v7": float(np.mean(ori_err_v7)),
        "ori_aeps_ins": float(np.mean(ori_err_ins)),
        "final_drift_v7": float(drift_v7),
        "final_drift_ins": float(drift_ins),
        "drift_pct_v7": (drift_v7 / total_dist * 100) if total_dist > 0 else 0.0,
        "drift_pct_ins": (drift_ins / total_dist * 100) if total_dist > 0 else 0.0,
        "drift_1s": drift_horizons.get("drift_1s", 0.0),
        "drift_3s": drift_horizons.get("drift_3s", 0.0),
        "drift_5s": drift_horizons.get("drift_5s", 0.0),
        "drift_10s": float(drift_v7),
        "pos_gt": np.array(pos_gt),
        "pos_v7": np.array(pos_v7),
        "pos_ins": np.array(pos_ins),
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_trajectory_v7(res: dict, scen_name: str, out_dir: Path, tag: str = "v7"):
    fig, ax = plt.subplots(figsize=(8, 7))
    gt = res["pos_gt"]
    v7 = res["pos_v7"]
    ins = res["pos_ins"]

    ax.plot(gt[:, 1], gt[:, 0], "g-", linewidth=2.5, label="Ground Truth GPS")
    ax.plot(v7[:, 1], v7[:, 0], "b--", linewidth=2.0, label=f"PINO-DR {tag.upper()} (Dual Specialist)")
    ax.plot(ins[:, 1], ins[:, 0], "r:", linewidth=1.8, label="Pure INS Dead Reckoning")

    ax.scatter([gt[0, 1]], [gt[0, 0]], color="black", marker="o", s=80, zorder=5, label="Outage Start")
    ax.scatter([gt[-1, 1]], [gt[-1, 0]], color="green", marker="X", s=90, zorder=5, label="GPS End")
    ax.scatter([v7[-1, 1]], [v7[-1, 0]], color="blue", marker="^", s=90, zorder=5, label=f"{tag.upper()} End")

    ax.set_title(
        f"10-s Outage: {scen_name.replace('_', ' ').title()}\n"
        f"{tag.upper()} Drift: {res['final_drift_v7']:.1f}m ({res['drift_pct_v7']:.1f}%) | "
        f"INS: {res['final_drift_ins']:.1f}m ({res['drift_pct_ins']:.1f}%) | Turn Blend: {res['mean_alpha']*100:.0f}% S2"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    path = out_dir / f"trajectory_{tag}_{scen_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ─── Full Evaluation Pipeline ────────────────────────────────────────────────

def evaluate_all(
    use_untuned: bool = False,
    tag: str = "v7",
    measure_mode: str = "production",
    s1_source: str = "tuned",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    switcher = load_v7_switcher(
        device,
        use_untuned_checkpoints=use_untuned,
        fusion_mode=measure_mode,
        s1_source=s1_source,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline benchmarks for comparison (v3 and v4 Ablation-D)
    v3_summary = {}
    v3_path = WS_ROOT / "v3_pino_dr" / "results" / "benchmark_summary_v3.json"
    if v3_path.exists():
        with open(v3_path, "r", encoding="utf-8") as f:
            v3_summary = json.load(f)

    v4_d_summary = {}
    v4_d_path = WS_ROOT / "v4_turn_focused" / "results" / "benchmark_summary_v4_ablation_D.json"
    if v4_d_path.exists():
        with open(v4_d_path, "r", encoding="utf-8") as f:
            v4_d_summary = json.load(f)

    summary = {}
    all_seq_results = []

    banner_title = f"PINO-DR v7 ({'UNTUNED BASELINE' if use_untuned else 'FINE-TUNED DUAL SPECIALIST'}, MODE={measure_mode.upper()})"
    print("\n" + "=" * 95)
    print(banner_title)
    print("=" * 95)

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
                res = simulate_sequence_v7(j, start, switcher, scalers, device, measure_mode=measure_mode)
                seq_results.append(res)
                all_seq_results.append(res)
                if sample is None and res["total_distance"] > 50:
                    sample = res

        n_seq = len(seq_results)
        keys = [
            "disp_crse_v7", "disp_crse_ins", "disp_aeps_v7", "disp_aeps_ins",
            "ori_crse_v7", "ori_crse_ins", "ori_aeps_v7", "ori_aeps_ins",
            "final_drift_v7", "final_drift_ins", "drift_pct_v7", "drift_pct_ins",
            "drift_1s", "drift_3s", "drift_5s", "drift_10s", "mean_alpha",
        ]
        rec = {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
        rec["n_sequences"] = n_seq
        rec["disp_improvement_vs_ins_pct"] = (
            (rec["disp_crse_ins"] - rec["disp_crse_v7"]) / rec["disp_crse_ins"] * 100
            if rec["disp_crse_ins"] > 0 else 0.0
        )
        rec["ori_improvement_vs_ins_pct"] = (
            (rec["ori_crse_ins"] - rec["ori_crse_v7"]) / rec["ori_crse_ins"] * 100
            if rec["ori_crse_ins"] > 0 else 0.0
        )

        high_yaw_seqs = [r for r in seq_results if r["is_high_yaw"]]
        low_yaw_seqs = [r for r in seq_results if not r["is_high_yaw"]]

        rec["high_yaw_drift"] = float(np.mean([r["final_drift_v7"] for r in high_yaw_seqs])) if high_yaw_seqs else None
        rec["low_yaw_drift"] = float(np.mean([r["final_drift_v7"] for r in low_yaw_seqs])) if low_yaw_seqs else None

        # Comparison with v3
        if scen_name in v3_summary:
            v3 = v3_summary[scen_name]
            rec["v3_drift"] = v3.get("final_drift_v3", None)
            if rec["v3_drift"]:
                rec["imprv_vs_v3_pct"] = (rec["v3_drift"] - rec["final_drift_v7"]) / rec["v3_drift"] * 100

        # Comparison with v4 Ablation-D
        if scen_name in v4_d_summary:
            v4d = v4_d_summary[scen_name]
            rec["v4_d_drift"] = v4d.get("final_drift_v4", None)
            if rec["v4_d_drift"]:
                rec["imprv_vs_v4d_pct"] = (rec["v4_d_drift"] - rec["final_drift_v7"]) / rec["v4_d_drift"] * 100

        summary[scen_name] = rec

        v3_str = f" | v3={rec.get('v3_drift', 0):.2f}m ({rec.get('imprv_vs_v3_pct', 0):+.1f}%)" if rec.get("v3_drift") else ""
        v4d_str = f" | v4D={rec.get('v4_d_drift', 0):.2f}m ({rec.get('imprv_vs_v4d_pct', 0):+.1f}%)" if rec.get("v4_d_drift") else ""
        print(f"\nScenario [{scen_name.upper()}] ({n_seq} sequences)")
        print(f"  Disp CRSE: {tag}={rec['disp_crse_v7']:.2f}m  INS={rec['disp_crse_ins']:.2f}m  (Imprv {rec['disp_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  Ori  CRSE: {tag}={rec['ori_crse_v7']:.3f}rad INS={rec['ori_crse_ins']:.3f}rad (Imprv {rec['ori_improvement_vs_ins_pct']:+.1f}%)")
        print(f"  10s Drift: {tag}={rec['final_drift_v7']:.2f}m ({rec['drift_pct_v7']:.1f}%)  INS={rec['final_drift_ins']:.2f}m{v3_str}{v4d_str}")
        print(f"  Horizons:  1s={rec['drift_1s']:.2f}m | 3s={rec['drift_3s']:.2f}m | 5s={rec['drift_5s']:.2f}m | 10s={rec['drift_10s']:.2f}m")
        print(f"  S2 Weight: {rec['mean_alpha']*100:.1f}% average turn-specialist activation")

        if sample is not None:
            p = plot_trajectory_v7(sample, scen_name, RESULTS_DIR, tag=tag)
            print(f"  [Plot] {p}")

    # ── Overall Metrics across All Sequences ──
    all_high_yaw = [r for r in all_seq_results if r["is_high_yaw"]]
    all_low_yaw = [r for r in all_seq_results if not r["is_high_yaw"]]

    overall = {
        "total_sequences": len(all_seq_results),
        "overall_drift_10s": float(np.mean([r["final_drift_v7"] for r in all_seq_results])),
        "overall_drift_ins": float(np.mean([r["final_drift_ins"] for r in all_seq_results])),
        "overall_horizons": {
            "drift_1s": float(np.mean([r["drift_1s"] for r in all_seq_results])),
            "drift_3s": float(np.mean([r["drift_3s"] for r in all_seq_results])),
            "drift_5s": float(np.mean([r["drift_5s"] for r in all_seq_results])),
            "drift_10s": float(np.mean([r["drift_10s"] for r in all_seq_results])),
        },
        "high_yaw": {
            "n_sequences": len(all_high_yaw),
            "drift_10s": float(np.mean([r["final_drift_v7"] for r in all_high_yaw])) if all_high_yaw else 0.0,
            "drift_ins": float(np.mean([r["final_drift_ins"] for r in all_high_yaw])) if all_high_yaw else 0.0,
            "ori_crse": float(np.mean([r["ori_crse_v7"] for r in all_high_yaw])) if all_high_yaw else 0.0,
        },
        "low_yaw": {
            "n_sequences": len(all_low_yaw),
            "drift_10s": float(np.mean([r["final_drift_v7"] for r in all_low_yaw])) if all_low_yaw else 0.0,
            "drift_ins": float(np.mean([r["final_drift_ins"] for r in all_low_yaw])) if all_low_yaw else 0.0,
            "ori_crse": float(np.mean([r["ori_crse_v7"] for r in all_low_yaw])) if all_low_yaw else 0.0,
        },
    }
    summary["overall_breakdown"] = overall

    print("\n" + "=" * 95)
    print("OVERALL SYSTEM SUMMARY (ALL TEST SEQUENCES):")
    print(f"  Overall 10-s Drift:     {tag} = {overall['overall_drift_10s']:.2f}m | INS = {overall['overall_drift_ins']:.2f}m")
    print(f"  Horizon Progression:    1s = {overall['overall_horizons']['drift_1s']:.2f}m | 3s = {overall['overall_horizons']['drift_3s']:.2f}m | 5s = {overall['overall_horizons']['drift_5s']:.2f}m | 10s = {overall['overall_horizons']['drift_10s']:.2f}m")
    print(f"  High-Yaw Regime ({overall['high_yaw']['n_sequences']} seqs): Drift = {overall['high_yaw']['drift_10s']:.2f}m | Ori CRSE = {overall['high_yaw']['ori_crse']:.3f}rad")
    print(f"  Low-Yaw Regime  ({overall['low_yaw']['n_sequences']} seqs): Drift = {overall['low_yaw']['drift_10s']:.2f}m | Ori CRSE = {overall['low_yaw']['ori_crse']:.3f}rad")
    print("=" * 95)

    # ── Latency Benchmark ──
    print("\n[v7] Running dual-specialist switcher latency benchmark (1,000 iterations)...")
    dummy = torch.randn(1, 10, 6, device=device)
    dummy_w = np.zeros(10, dtype=np.float32)
    dummy_a = np.zeros(10, dtype=np.float32)
    for _ in range(50):
        with torch.no_grad():
            if isinstance(switcher, ProductionDualSpecialistFusion):
                switcher(dummy, dummy_w, dummy_a, 15.0)
            else:
                switcher(dummy)

    import time
    t0 = time.perf_counter()
    n_iter = 1000
    for _ in range(n_iter):
        with torch.no_grad():
            if isinstance(switcher, ProductionDualSpecialistFusion):
                switcher(dummy, dummy_w, dummy_a, 15.0)
            else:
                switcher(dummy)
    avg_ms = (time.perf_counter() - t0) * 1000 / n_iter
    print(f"[v7] Latency: {avg_ms:.3f} ms/step ({n_iter} iterations on {device})")
    summary["latency_ms"] = avg_ms

    # ── Save JSON ──
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

    out_file = RESULTS_DIR / f"benchmark_summary_{tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"[v7] Benchmark saved -> {out_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dual-Specialist PINO-DR v7")
    parser.add_argument("--untuned", action="store_true", help="Evaluate untuned v3 + v4-D base models")
    parser.add_argument("--tag", default="v7_production", help="Output identifier tag (e.g. v7_production, v7_highway_anchor, etc.)")
    parser.add_argument("--mode", choices=["production", "standard", "adaptive"], default="production", help="Fusion architecture mode")
    parser.add_argument("--s1_source", choices=["tuned", "v3"], default="tuned", help="Straight specialist weights source (tuned or v3 base)")
    args = parser.parse_args()

    evaluate_all(use_untuned=args.untuned, tag=args.tag, measure_mode=args.mode, s1_source=args.s1_source)
