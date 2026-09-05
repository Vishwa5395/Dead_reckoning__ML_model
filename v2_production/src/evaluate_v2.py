"""
evaluate_v2.py
--------------
Closed-loop, autoregressive benchmark evaluation for the v2 IDNN models.

This mirrors the legacy evaluate.py protocol (10-second GNSS outages on the
named test scenarios), but:
- Uses the v2 models (checkpoints_v2/) and the CLEAN scalers.
- Clips model predictions to the physical bounds used during training so the
  closed-loop feedback never leaves the learned distribution.
- Writes results/benchmark_summary_v2.json, trajectory plots, and MAE curves.

AUTOREGRESSIVE CLOSED-LOOP:
The displacement model feeds back its OWN previous prediction x_pred(t-1)
(never ground-truth GPS) during the outage, exactly as in the paper.
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

from src.models_v2 import DisplacementIDNNV2, OrientationIDNNV2

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "preprocessed" / "clean"
CKPT_DIR = ROOT / "checkpoints_v2"
RESULTS_DIR = ROOT / "results"

# Physical bounds (must match src/preprocess_v2.py CLIP + train_v2.py DISP/ORI_BOUNDS)
DISP_LO, DISP_HI = 0.0, 45.0
ORI_LO, ORI_HI = -1.2, 1.2

WINDOW = 10
OUTAGE = 10

SCENARIO_ORDER = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]


def load_v2_models(scalers: dict, device: torch.device):
    disp = DisplacementIDNNV2(input_dim=20, hidden_dims=(64, 64, 32), dropout=0.20).to(device)
    ori = OrientationIDNNV2(input_dim=10, hidden_dims=(64, 64, 32), dropout=0.20).to(device)
    d_ckpt = torch.load(CKPT_DIR / "displacement_idnn.pth", map_location=device)
    o_ckpt = torch.load(CKPT_DIR / "orientation_idnn.pth", map_location=device)
    disp.load_state_dict(d_ckpt["model_state_dict"])
    ori.load_state_dict(o_ckpt["model_state_dict"])
    disp.eval()
    ori.eval()
    return disp, ori


def simulate_sequence(journey, start_idx, disp_model, ori_model, scalers, device):
    a_ins = journey["a_ins"]
    w_ins = journey["w_ins"]
    x_gps = journey["x_gps"]
    w_gps = journey["w_gps"]
    headings = journey["headings"]

    s_dX = scalers["disp_X"]
    s_dY = scalers["disp_Y"]
    s_oX = scalers["ori_X"]
    s_oY = scalers["ori_Y"]

    history_x_pred = list(x_gps[start_idx - WINDOW : start_idx])

    pos_gt = [(0.0, 0.0)]
    pos_idnn = [(0.0, 0.0)]
    pos_ins = [(0.0, 0.0)]

    psi_gt = np.radians(headings[start_idx])
    psi_idnn = psi_gt
    psi_ins = psi_gt
    v_ins = x_gps[start_idx - 1]

    disp_err = []
    ins_err = []
    ori_err_idnn = []
    ori_err_ins = []

    for k in range(OUTAGE):
        cur = start_idx + k

        # Orientation prediction
        w_win = w_ins[cur - WINDOW + 1 : cur + 1].reshape(1, -1)
        w_win_scaled = s_oX.transform(w_win)
        with torch.no_grad():
            w_pred_scaled = ori_model(torch.tensor(w_win_scaled, dtype=torch.float32).to(device)).item()
        w_raw = s_oY.inverse_transform([[w_pred_scaled]])[0, 0]
        w_pred = float(np.clip(w_raw, ORI_LO, ORI_HI))

        true_w = w_gps[cur]
        raw_w = w_ins[cur]
        psi_gt += true_w
        psi_idnn += w_pred
        psi_ins += raw_w
        ori_err_idnn.append(abs(true_w - w_pred))
        ori_err_ins.append(abs(true_w - raw_w))

        # Displacement prediction (autoregressive feedback)
        a_win = a_ins[cur - WINDOW + 1 : cur + 1]
        x_win = np.array(history_x_pred[-WINDOW:])
        feat = np.concatenate([a_win, x_win]).reshape(1, -1)
        feat_scaled = s_dX.transform(feat)
        with torch.no_grad():
            x_scaled = disp_model(torch.tensor(feat_scaled, dtype=torch.float32).to(device)).item()
        x_raw = s_dY.inverse_transform([[x_scaled]])[0, 0]
        x_pred = float(np.clip(x_raw, DISP_LO, DISP_HI))
        history_x_pred.append(x_pred)

        # INS baseline
        v_ins = max(0.0, v_ins + a_ins[cur])
        x_ins = v_ins

        true_x = x_gps[cur]
        disp_err.append(abs(true_x - x_pred))
        ins_err.append(abs(true_x - x_ins))

        # Position updates (NED)
        dn_gt = true_x * np.cos(psi_gt)
        de_gt = true_x * np.sin(psi_gt)
        pos_gt.append((pos_gt[-1][0] + dn_gt, pos_gt[-1][1] + de_gt))

        dn_idnn = x_pred * np.cos(psi_idnn)
        de_idnn = x_pred * np.sin(psi_idnn)
        pos_idnn.append((pos_idnn[-1][0] + dn_idnn, pos_idnn[-1][1] + de_idnn))

        dn_ins = x_ins * np.cos(psi_ins)
        de_ins = x_ins * np.sin(psi_ins)
        pos_ins.append((pos_ins[-1][0] + dn_ins, pos_ins[-1][1] + de_ins))

    total_dist = sum(x_gps[start_idx : start_idx + OUTAGE])
    drift_idnn = np.hypot(pos_idnn[-1][0] - pos_gt[-1][0], pos_idnn[-1][1] - pos_gt[-1][1])
    drift_ins = np.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1])

    return {
        "total_distance": total_dist,
        "disp_crse_idnn": sum(disp_err),
        "disp_crse_ins": sum(ins_err),
        "disp_aeps_idnn": float(np.mean(disp_err)),
        "disp_aeps_ins": float(np.mean(ins_err)),
        "ori_crse_idnn": sum(ori_err_idnn),
        "ori_crse_ins": sum(ori_err_ins),
        "ori_aeps_idnn": float(np.mean(ori_err_idnn)),
        "ori_aeps_ins": float(np.mean(ori_err_ins)),
        "final_drift_idnn": float(drift_idnn),
        "final_drift_ins": float(drift_ins),
        "drift_pct_idnn": (drift_idnn / total_dist * 100.0) if total_dist > 0 else 0.0,
        "drift_pct_ins": (drift_ins / total_dist * 100.0) if total_dist > 0 else 0.0,
        "pos_gt": np.array(pos_gt),
        "pos_idnn": np.array(pos_idnn),
        "pos_ins": np.array(pos_ins),
    }


def plot_trajectory(res, scen_name, results_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    gt = res["pos_gt"]
    idnn = res["pos_idnn"]
    ins = res["pos_ins"]
    ax.plot(gt[:, 1], gt[:, 0], "g-", linewidth=2.5, label="Ground Truth GPS")
    ax.plot(idnn[:, 1], idnn[:, 0], "b--", linewidth=2.0, label="IDNN v2 (Closed-Loop)")
    ax.plot(ins[:, 1], ins[:, 0], "r:", linewidth=1.8, label="Pure INS Dead Reckoning")
    ax.scatter([gt[0, 1]], [gt[0, 0]], color="black", marker="o", s=80, zorder=5, label="Outage Start")
    ax.scatter([gt[-1, 1]], [gt[-1, 0]], color="green", marker="X", s=90, zorder=5, label="GPS End")
    ax.scatter([idnn[-1, 1]], [idnn[-1, 0]], color="blue", marker="^", s=90, zorder=5, label="IDNN End")
    ax.set_title(
        f"10-Sec GNSS Outage: {scen_name.replace('_',' ').title()}\n"
        f"v2 IDNN Drift: {res['final_drift_idnn']:.1f}m ({res['drift_pct_idnn']:.1f}%) vs "
        f"INS: {res['final_drift_ins']:.1f}m ({res['drift_pct_ins']:.1f}%)"
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    path = results_dir / f"trajectory_v2_{scen_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


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


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CACHE_DIR / "scalers_clean.pkl", "rb") as f:
        scalers = pickle.load(f)
    # test scenarios are journey-level raw data stored in the ORIGINAL cache
    with open(ROOT / "data" / "preprocessed" / "test_scenarios.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    disp_model, ori_model = load_v2_models(scalers, device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    print("\n" + "=" * 80)
    print("IO-VNBD v2 BENCHMARK (AUTOREGRESSIVE CLOSED-LOOP, CLEAN SCALERS)")
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
                res = simulate_sequence(j, start, disp_model, ori_model, scalers, device)
                seq_results.append(res)
                if sample is None and res["total_distance"] > 50:
                    sample = res

        n_seq = len(seq_results)
        keys = [
            "disp_crse_idnn", "disp_crse_ins", "disp_aeps_idnn", "disp_aeps_ins",
            "ori_crse_idnn", "ori_crse_ins", "ori_aeps_idnn", "ori_aeps_ins",
            "final_drift_idnn", "final_drift_ins", "drift_pct_idnn", "drift_pct_ins",
        ]
        rec = {k: float(np.mean([r[k] for r in seq_results])) for k in keys}
        rec["n_sequences"] = n_seq
        rec["disp_improvement_pct"] = (
            (rec["disp_crse_ins"] - rec["disp_crse_idnn"]) / rec["disp_crse_ins"] * 100.0
            if rec["disp_crse_ins"] > 0 else 0.0
        )
        rec["ori_improvement_pct"] = (
            (rec["ori_crse_ins"] - rec["ori_crse_idnn"]) / rec["ori_crse_ins"] * 100.0
            if rec["ori_crse_ins"] > 0 else 0.0
        )
        summary[scen_name] = rec

        print(f"\nScenario [{scen_name.upper()}] ({n_seq} sequences)")
        print(f"  Disp CRSE: IDNN={rec['disp_crse_idnn']:.2f}m INS={rec['disp_crse_ins']:.2f}m "
              f"(Imprv {rec['disp_improvement_pct']:+.1f}%)")
        print(f"  Ori CRSE:  IDNN={rec['ori_crse_idnn']:.3f}rad/s INS={rec['ori_crse_ins']:.3f}rad/s "
              f"(Imprv {rec['ori_improvement_pct']:+.1f}%)")
        print(f"  10s Drift: IDNN={rec['final_drift_idnn']:.2f}m ({rec['drift_pct_idnn']:.2f}%) "
              f"INS={rec['final_drift_ins']:.2f}m ({rec['drift_pct_ins']:.2f}%)")

        if sample is not None:
            p = plot_trajectory(sample, scen_name, RESULTS_DIR)
            print(f"  [Plot] {p}")

    with open(RESULTS_DIR / "benchmark_summary_v2.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    with open(RESULTS_DIR / "benchmark_summary_v2.pkl", "wb") as f:
        pickle.dump(summary, f)

    plot_mae_curves(summary, RESULTS_DIR)
    print(f"\n[v2] Benchmark summary saved: {RESULTS_DIR/'benchmark_summary_v2.json'}")
    return summary


def plot_mae_curves(summary, results_dir):
    names = [s for s in SCENARIO_ORDER if s in summary]
    if not names:
        return
    disp_idnn = [summary[s]["disp_crse_idnn"] for s in names]
    disp_ins = [summary[s]["disp_crse_ins"] for s in names]
    ori_idnn = [summary[s]["ori_crse_idnn"] for s in names]
    ori_ins = [summary[s]["ori_crse_ins"] for s in names]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(names, disp_idnn, marker="o", label="v2 IDNN displacement", color="tab:blue")
    axes[0].plot(names, disp_ins, marker="s", label="INS displacement", color="tab:orange", linestyle="--")
    axes[0].set_title("Displacement MAE (CRSE) by scenario (v2)")
    axes[0].set_ylabel("MAE (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")

    axes[1].plot(names, ori_idnn, marker="o", label="v2 IDNN orientation", color="tab:green")
    axes[1].plot(names, ori_ins, marker="s", label="INS orientation", color="tab:red", linestyle="--")
    axes[1].set_title("Orientation MAE (CRSE) by scenario (v2)")
    axes[1].set_ylabel("MAE (rad/s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    out = results_dir / "mae_curve_v2.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[v2] MAE curves saved: {out}")


if __name__ == "__main__":
    evaluate_all()
