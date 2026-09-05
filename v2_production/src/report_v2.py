"""
report_v2.py
------------
Generates a comprehensive production report for the v2 model and compares it
to the previous (legacy) model.

Uses:
- results/benchmark_summary.json      (previous model benchmark)
- results/benchmark_summary_v2.json   (v2 model benchmark)
- results/training_history_{disp,ori}.json (per-epoch train/val MAE)
Writes:
- results/model_quality_report_v2.txt / .json
- results/training_curves_v2.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]


def load_json(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report() -> dict:
    old = load_json("benchmark_summary.json")
    new = load_json("benchmark_summary_v2.json")

    disp_hist = load_json("training_history_displacement.json")
    ori_hist = load_json("training_history_orientation.json")

    rows = []
    for s in SCENARIOS:
        if s not in old or s not in new:
            continue
        o, n = old[s], new[s]
        rows.append(
            {
                "scenario": s,
                "old_disp_mae_idnn": o["disp_crse_idnn"],
                "new_disp_mae_idnn": n["disp_crse_idnn"],
                "disp_mae_ins": n["disp_crse_ins"],
                "disp_delta_pct": (n["disp_crse_idnn"] - o["disp_crse_idnn"]) / o["disp_crse_idnn"] * 100.0
                if o["disp_crse_idnn"] > 0 else 0.0,
                "disp_impr_vs_ins": n["disp_improvement_pct"],
                "old_disp_impr_vs_ins": o["disp_improvement_pct"],
                "old_ori_mae_idnn": o["ori_crse_idnn"],
                "new_ori_mae_idnn": n["ori_crse_idnn"],
                "ori_mae_ins": n["ori_crse_ins"],
                "ori_delta_pct": (n["ori_crse_idnn"] - o["ori_crse_idnn"]) / o["ori_crse_idnn"] * 100.0
                if o["ori_crse_idnn"] > 0 else 0.0,
                "ori_impr_vs_ins": n["ori_improvement_pct"],
                "old_drift_pct": o["drift_pct_idnn"],
                "new_drift_pct": n["drift_pct_idnn"],
                "new_drift_m": n["final_drift_idnn"],
                "new_n_seq": n["n_sequences"],
            }
        )

    avg_delta_disp = sum(r["disp_delta_pct"] for r in rows) / len(rows)
    avg_delta_ori = sum(r["ori_delta_pct"] for r in rows) / len(rows)
    avg_new_impr_disp = sum(r["disp_impr_vs_ins"] for r in rows) / len(rows)
    avg_new_impr_ori = sum(r["ori_impr_vs_ins"] for r in rows) / len(rows)
    n_better_disp = sum(1 for r in rows if r["disp_delta_pct"] < 0)
    n_better_ori = sum(1 for r in rows if r["ori_delta_pct"] < 0)

    # Aggregate training metrics
    d_best = disp_hist.get("best_val_mae")
    d_test = disp_hist.get("test_mae")
    d_epochs = len(disp_hist.get("history", []))
    o_best = ori_hist.get("best_val_mae")
    o_test = ori_hist.get("test_mae")
    o_epochs = len(ori_hist.get("history", []))

    d_hist_list = disp_hist.get("history", [])
    o_hist_list = ori_hist.get("history", [])
    d_train_min = min(h["train_mae"] for h in d_hist_list) if d_hist_list else None
    d_val_min = min(h["val_mae"] for h in d_hist_list) if d_hist_list else None
    o_train_min = min(h["train_mae"] for h in o_hist_list) if o_hist_list else None
    o_val_min = min(h["val_mae"] for h in o_hist_list) if o_hist_list else None

    return {
        "summary": {
            "scenario_count": len(rows),
            "old_avg_disp_impr_pct": sum(r["old_disp_impr_vs_ins"] for r in rows) / len(rows) if rows else 0,
            "avg_new_disp_impr_vs_ins_pct": avg_new_impr_disp,
            "avg_new_ori_impr_vs_ins_pct": avg_new_impr_ori,
            "avg_disp_mae_reduction_pct": avg_delta_disp,
            "avg_ori_mae_reduction_pct": avg_delta_ori,
            "scenarios_disp_better_than_old": n_better_disp,
            "scenarios_ori_better_than_old": n_better_ori,
            "displacement_training": {
                "best_val_mae": d_best, "test_mae": d_test, "epochs_run": d_epochs,
                "train_mae_min": d_train_min, "val_mae_min": d_val_min,
            },
            "orientation_training": {
                "best_val_mae": o_best, "test_mae": o_test, "epochs_run": o_epochs,
                "train_mae_min": o_train_min, "val_mae_min": o_val_min,
            },
            "status": (
                "production_ready" if (avg_new_impr_disp > 0 and avg_new_impr_ori > 0 and avg_delta_disp < 0)
                else "improved_but_remaining_gaps"
            ),
        },
        "scenarios": rows,
    }


def write_text(report: dict) -> None:
    s = report["summary"]
    lines = []
    lines.append("MODEL QUALITY REPORT — v2 (PRODUCTION) vs PREVIOUS MODEL")
    lines.append("=" * 90)
    lines.append(f"Status: {s['status']}")
    lines.append(f"Scenarios evaluated: {s['scenario_count']}")
    lines.append("")
    lines.append("SUMMARY METRICS")
    lines.append("-" * 90)
    lines.append(f"Avg displacement improvement vs INS:   OLD = {s['old_avg_disp_impr_pct']:+.2f}%  ->  v2 = {s['avg_new_disp_impr_vs_ins_pct']:+.2f}%")
    lines.append(f"Avg orientation improvement vs INS:   v2 = {s['avg_new_ori_impr_vs_ins_pct']:+.2f}%")
    lines.append(f"Avg displacement MAE change (v2 vs old): {s['avg_disp_mae_reduction_pct']:+.2f}%  (negative = v2 better)")
    lines.append(f"Avg orientation MAE change (v2 vs old): {s['avg_ori_mae_reduction_pct']:+.2f}%  (negative = v2 better)")
    lines.append(f"Scenarios where v2 displacement MAE improved: {s['scenarios_disp_better_than_old']}/{s['scenario_count']}")
    lines.append(f"Scenarios where v2 orientation MAE improved:  {s['scenarios_ori_better_than_old']}/{s['scenario_count']}")
    lines.append("")
    lines.append("TRAINING (held-out 20% test) — v2")
    lines.append("-" * 90)
    dt = s["displacement_training"]
    ot = s["orientation_training"]
    lines.append(f"Displacement: best val MAE={dt['best_val_mae']:.5f}  test MAE={dt['test_mae']:.5f}  epochs={dt['epochs_run']}")
    lines.append(f"  train MAE min={dt['train_mae_min']:.5f}  val MAE min={dt['val_mae_min']:.5f}")
    lines.append(f"Orientation:  best val MAE={ot['best_val_mae']:.5f}  test MAE={ot['test_mae']:.5f}  epochs={ot['epochs_run']}")
    lines.append(f"  train MAE min={ot['train_mae_min']:.5f}  val MAE min={ot['val_mae_min']:.5f}")
    lines.append("")
    lines.append("SCENARIO-BY-SCENARIO COMPARISON (CRSE = 10-s cumulative error)")
    lines.append("-" * 90)
    hdr = f"{'Scenario':<12} | {'old Disp':>9} | {'new Disp':>9} | {'Δ%':>8} | {'Disp INS':>9} | {'old Ori':>8} | {'new Ori':>8} | {'Ori Δ%':>8} | {'Ori INS':>8} | {'old Drift%':>10} | {'new Drift%':>10}"
    lines.append(hdr)
    lines.append("-" * 90)
    for r in report["scenarios"]:
        lines.append(
            f"{r['scenario']:<12} | {r['old_disp_mae_idnn']:>9.2f} | {r['new_disp_mae_idnn']:>9.2f} | "
            f"{r['disp_delta_pct']:>+8.1f} | {r['disp_mae_ins']:>9.2f} | {r['old_ori_mae_idnn']:>8.3f} | "
            f"{r['new_ori_mae_idnn']:>8.3f} | {r['ori_delta_pct']:>+8.1f} | {r['ori_mae_ins']:>8.3f} | "
            f"{r['old_drift_pct']:>10.1f} | {r['new_drift_pct']:>10.1f}"
        )
    lines.append("")
    lines.append("ROOT CAUSE FIX APPLIED")
    lines.append("-" * 90)
    lines.append("The previous model's displacement target was scaled by a MinMaxScaler fit to a")
    lines.append("catastrophic GPS outlier (max 1843.7 m/s for a single 1-second step, vs a 99.99th")
    lines.append("percentile of ~36 m/s). This collapsed all real driving data to ~0.006 on [0,1], so")
    lines.append("the model learned a near-constant '~12 m/s' predictor. v2 reconstructs raw values,")
    lines.append("clips outliers to physical road-vehicle limits ([0,45] m/s, accel [-8,8] m/s^2,")
    lines.append("gyro [-1,1] rad/s), and re-fits scalers on the train split only. v2 also upgrades the")
    lines.append("network (3 hidden layers 64-64-32, BatchNorm, GELU) and training (AdamW, LR scheduling,")
    lines.append("early stopping, gradient clipping, seeds).")

    out = RESULTS / "model_quality_report_v2.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[v2] Text report written: {out}")


def write_json(report: dict) -> None:
    out = RESULTS / "model_quality_report_v2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[v2] JSON report written: {out}")


def plot_training_curves() -> None:
    try:
        d = load_json("training_history_displacement.json")["history"]
        o = load_json("training_history_orientation.json")["history"]
    except Exception:
        print("[v2] Training history not found; skipping curve plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ep_d = [h["epoch"] for h in d]
    axes[0].plot(ep_d, [h["train_mae"] for h in d], label="Train MAE", marker="o", markersize=2)
    axes[0].plot(ep_d, [h["val_mae"] for h in d], label="Val MAE", marker="x", markersize=2)
    axes[0].set_title("Displacement IDNN v2 — train/val MAE")
    axes[0].set_ylabel("MAE (scaled)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    ep_o = [h["epoch"] for h in o]
    axes[1].plot(ep_o, [h["train_mae"] for h in o], label="Train MAE", marker="o", markersize=2)
    axes[1].plot(ep_o, [h["val_mae"] for h in o], label="Val MAE", marker="x", markersize=2)
    axes[1].set_title("Orientation IDNN v2 — train/val MAE")
    axes[1].set_ylabel("MAE (scaled)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    out = RESULTS / "training_curves_v2.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[v2] Training curves saved: {out}")


def main() -> None:
    report = build_report()
    write_text(report)
    write_json(report)
    plot_training_curves()


if __name__ == "__main__":
    main()
