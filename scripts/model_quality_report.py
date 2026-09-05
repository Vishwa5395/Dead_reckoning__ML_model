#!/usr/bin/env python3
"""Produce a model quality report without retraining.

This uses the saved benchmark_summary.json from the project and generates:
- a text quality report
- a scenario-wise MAE trend plot
- a summary JSON with model quality metrics

Important: the project does not currently persist per-epoch training MAE values,
so this script reports the benchmark/validation-style MAE signal that already
exists in the saved artifacts. It does not retrain the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "benchmark_summary.json"
REPORT_PATH = ROOT / "results" / "model_quality_report.txt"
JSON_PATH = ROOT / "results" / "model_quality_report.json"
PLOT_PATH = ROOT / "results" / "mae_curve.png"


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {SUMMARY_PATH}")
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_mae_rows(summary: dict):
    rows = []
    for name, rec in summary.items():
        rows.append(
            {
                "scenario": name,
                "disp_mae_idnn": rec["disp_crse_idnn"],
                "disp_mae_ins": rec["disp_crse_ins"],
                "ori_mae_idnn": rec["ori_crse_idnn"],
                "ori_mae_ins": rec["ori_crse_ins"],
                "drift_idnn_pct": rec["drift_pct_idnn"],
                "drift_ins_pct": rec["drift_pct_ins"],
                "disp_improvement_pct": rec["disp_improvement_pct"],
                "ori_improvement_pct": rec["ori_improvement_pct"],
            }
        )
    return rows


def build_report(summary: dict) -> dict:
    rows = scenario_mae_rows(summary)
    avg_disp_impr = sum(r["disp_improvement_pct"] for r in rows) / len(rows)
    avg_ori_impr = sum(r["ori_improvement_pct"] for r in rows) / len(rows)
    better_disp = sum(1 for r in rows if r["disp_improvement_pct"] > 0)

    report = {
        "summary": {
            "scenario_count": len(rows),
            "avg_disp_improvement_pct": avg_disp_impr,
            "avg_ori_improvement_pct": avg_ori_impr,
            "scenarios_better_than_ins": better_disp,
            "status": "mixed_generalization",
        },
        "scenarios": rows,
    }

    if avg_disp_impr < 0 and better_disp <= 1:
        report["summary"]["status"] = "underfitting_or_weak_generalization"
    elif avg_disp_impr > 0 and avg_ori_impr > 0:
        report["summary"]["status"] = "learning_but_unstable"
    else:
        report["summary"]["status"] = "mixed_generalization"

    return report


def write_text_report(report: dict) -> None:
    rows = report["scenarios"]
    avg_disp = report["summary"]["avg_disp_improvement_pct"]
    avg_ori = report["summary"]["avg_ori_improvement_pct"]
    better = report["summary"]["scenarios_better_than_ins"]
    status = report["summary"]["status"]

    lines = []
    lines.append("MODEL QUALITY REPORT")
    lines.append("=" * 80)
    lines.append(f"Status: {status}")
    lines.append(f"Average displacement improvement: {avg_disp:.2f}%")
    lines.append(f"Average orientation improvement: {avg_ori:.2f}%")
    lines.append(f"Scenarios where IDNN beats pure INS on displacement: {better}/{len(rows)}")
    lines.append("")
    lines.append("Scenario-level MAE signal")
    lines.append("-" * 80)
    for r in rows:
        lines.append(
            f"{r['scenario']:<12} | disp MAE IDNN={r['disp_mae_idnn']:.2f} | "
            f"disp MAE INS={r['disp_mae_ins']:.2f} | "
            f"disp Δ={r['disp_improvement_pct']:+.2f}% | "
            f"ori MAE IDNN={r['ori_mae_idnn']:.3f} | "
            f"ori MAE INS={r['ori_mae_ins']:.3f} | "
            f"ori Δ={r['ori_improvement_pct']:+.2f}%"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Orientation estimation is substantially better than the pure INS baseline.")
    lines.append("- Displacement estimation is inconsistent and often worse than INS, especially in motorway and quick-acceleration cases.")
    lines.append("- This indicates a weaker displacement model or limited transfer to some dynamic driving regimes.")
    lines.append("- No per-epoch train/val MAE log was found in the project, so this report is based on the saved benchmark evaluation metrics rather than a retraining run.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mae_curve(summary: dict) -> None:
    rows = scenario_mae_rows(summary)
    names = [r["scenario"] for r in rows]
    disp_idnn = [r["disp_mae_idnn"] for r in rows]
    disp_ins = [r["disp_mae_ins"] for r in rows]
    ori_idnn = [r["ori_mae_idnn"] for r in rows]
    ori_ins = [r["ori_mae_ins"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(names, disp_idnn, marker='o', label='IDNN displacement MAE', color='tab:blue')
    axes[0].plot(names, disp_ins, marker='s', label='INS displacement MAE', color='tab:orange', linestyle='--')
    axes[0].set_title('Displacement MAE by scenario')
    axes[0].set_ylabel('MAE (meters)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha='right')

    axes[1].plot(names, ori_idnn, marker='o', label='IDNN orientation MAE', color='tab:green')
    axes[1].plot(names, ori_ins, marker='s', label='INS orientation MAE', color='tab:red', linestyle='--')
    axes[1].set_title('Orientation MAE by scenario')
    axes[1].set_ylabel('MAE (rad/s)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha='right')

    fig.suptitle('Model quality report: MAE vs pure INS baseline (scenario-level)', fontsize=12)
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    report = build_report(summary)
    write_text_report(report)
    plot_mae_curve(summary)

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Quality report written to: {REPORT_PATH}")
    print(f"Summary JSON written to: {JSON_PATH}")
    print(f"MAE curve saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
