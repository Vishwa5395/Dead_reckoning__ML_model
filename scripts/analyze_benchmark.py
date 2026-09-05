#!/usr/bin/env python3
"""Summarize benchmark results and diagnose likely fit problems.

This reads the benchmark_summary.json produced by the benchmark script and
prints a concise breakdown of errors by scenario, along with a heuristic
classification for underfitting / overfitting based on the observed model
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def summarize():
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark summary not found: {summary_path}")

    summary = load_summary(summary_path)

    print("=" * 90)
    print("MODEL BENCHMARK DIAGNOSTIC REPORT")
    print("=" * 90)

    scenario_rows = []
    for name, rec in summary.items():
        disp_impr = rec["disp_improvement_pct"]
        ori_impr = rec["ori_improvement_pct"]
        drift_idnn = rec["drift_pct_idnn"]
        drift_ins = rec["drift_pct_ins"]
        scenario_rows.append(
            {
                "scenario": name,
                "disp_idnn": rec["disp_crse_idnn"],
                "disp_ins": rec["disp_crse_ins"],
                "disp_impr": disp_impr,
                "ori_idnn": rec["ori_crse_idnn"],
                "ori_ins": rec["ori_crse_ins"],
                "ori_impr": ori_impr,
                "drift_idnn": drift_idnn,
                "drift_ins": drift_ins,
            }
        )

    print("Scenario | Disp IDNN | Disp INS | Disp Δ | Ori IDNN | Ori INS | Ori Δ | Drift IDNN | Drift INS")
    for row in scenario_rows:
        print(
            f"{row['scenario']:>12} | "
            f"{row['disp_idnn']:>9.2f} | {row['disp_ins']:>8.2f} | {row['disp_impr']:>+7.2f}% | "
            f"{row['ori_idnn']:>8.3f} | {row['ori_ins']:>7.3f} | {row['ori_impr']:>+7.2f}% | "
            f"{row['drift_idnn']:>10.2f}% | {row['drift_ins']:>9.2f}%"
        )

    print("\n" + "-" * 90)
    print("ERROR INTERPRETATION")
    print("-" * 90)
    for row in scenario_rows:
        if row["disp_impr"] > 0:
            status = "displacement improvement"
        elif row["disp_impr"] > -15:
            status = "roughly neutral"
        else:
            status = "displacement failure"

        if row["ori_impr"] > 50:
            ori_status = "strongly learned"
        elif row["ori_impr"] > 0:
            ori_status = "moderately learned"
        else:
            ori_status = "not improved"

        print(
            f"{row['scenario']:<12} -> displacement: {status}; orientation: {ori_status}; "
            f"drift: IDNN {row['drift_idnn']:.2f}% vs INS {row['drift_ins']:.2f}%"
        )

    print("\n" + "-" * 90)
    print("FIT DIAGNOSTIC (HEURISTIC)")
    print("-" * 90)

    n_positive_disp = sum(1 for row in scenario_rows if row["disp_impr"] > 0)
    n_total = len(scenario_rows)
    avg_disp_impr = sum(row["disp_impr"] for row in scenario_rows) / n_total
    avg_ori_impr = sum(row["ori_impr"] for row in scenario_rows) / n_total

    print(f"Average displacement improvement: {avg_disp_impr:.2f}%")
    print(f"Average orientation improvement: {avg_ori_impr:.2f}%")
    print(f"Scenarios with better displacement than INS: {n_positive_disp}/{n_total}")

    if avg_disp_impr < 0 and n_positive_disp <= 1:
        print("Likely diagnosis: UNDERFITTING in displacement estimation.")
        print("Reason: the model fails to beat the baseline in most scenarios and the displacement CRSE is consistently high.")
    elif avg_disp_impr > 0 and avg_ori_impr > 0:
        print("Likely diagnosis: overall model is learning, but still unstable in hard dynamic regimes.")
        print("Reason: orientation improves strongly, while displacement improvement remains uneven across conditions.")
    else:
        print("Likely diagnosis: MIXED GENERALIZATION / PARTIAL OVERFIT.")
        print("Reason: the model learns heading well but displacement performance is inconsistent across scenarios, which indicates weak transfer to certain motion patterns.")

    print("\nNote: strict overfitting diagnosis requires the training-validation loss curves. This project does not currently save those curves, so this assessment is based on benchmark behavior and scenario-level generalization patterns.")


if __name__ == "__main__":
    summarize()
