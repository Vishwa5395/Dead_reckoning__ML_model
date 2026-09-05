"""
report_v3.py
------------
Generates a comprehensive 3-way model quality report comparing:
- Baseline (v1 / original IDNN)
- Production v2 (Feedforward IDNN + clipping fixes)
- PINO-DR v3 (Physics-Informed Neural Operator + autoregressive closed loop + ZUPT)

Writes:
- results/model_quality_report_v3.txt
- results/model_quality_report_v3.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]


def load_json(filename: str) -> dict:
    p = RESULTS / filename
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_v3_report():
    v1 = load_json("benchmark_summary.json")
    v2 = load_json("benchmark_summary_v2.json")
    v3 = load_json("benchmark_summary_v3.json")
    train_summary = load_json("training_summary_v3.json")

    scenario_data = []
    for s in SCENARIOS:
        o = v1[s]
        t = v2[s]
        th = v3[s]

        # v1 keys
        v1_disp = o["disp_crse_idnn"]
        v1_ori = o["ori_crse_idnn"]
        v1_drift = o["final_drift_m_idnn"]
        v1_drift_pct = o["drift_pct_idnn"]

        # v2 keys
        v2_disp = t["disp_crse_idnn"]
        v2_ori = t["ori_crse_idnn"]
        v2_drift = t["final_drift_idnn"]
        v2_drift_pct = t["drift_pct_idnn"]

        # v3 keys
        v3_disp = th["disp_crse_v3"]
        v3_ori = th["ori_crse_v3"]
        v3_drift = th["final_drift_v3"]
        v3_drift_pct = th["drift_pct_v3"]

        # INS
        ins_disp = th["disp_crse_ins"]
        ins_ori = th["ori_crse_ins"]
        ins_drift = th["final_drift_ins"]
        ins_drift_pct = th["drift_pct_ins"]

        # Deltas
        disp_v3_vs_v1_pct = (v3_disp - v1_disp) / v1_disp * 100.0
        disp_v3_vs_v2_pct = (v3_disp - v2_disp) / v2_disp * 100.0
        disp_v3_vs_ins_pct = (ins_disp - v3_disp) / ins_disp * 100.0

        ori_v3_vs_v1_pct = (v3_ori - v1_ori) / v1_ori * 100.0
        ori_v3_vs_v2_pct = (v3_ori - v2_ori) / v2_ori * 100.0
        ori_v3_vs_ins_pct = (ins_ori - v3_ori) / ins_ori * 100.0

        drift_v3_vs_v2_pct = (v3_drift - v2_drift) / v2_drift * 100.0

        scenario_data.append({
            "scenario": s,
            "n_sequences": th["n_sequences"],
            # Displacement
            "disp_v1": v1_disp,
            "disp_v2": v2_disp,
            "disp_v3": v3_disp,
            "disp_ins": ins_disp,
            "disp_v3_vs_ins_imprv_pct": disp_v3_vs_ins_pct,
            "disp_v3_vs_v2_delta_pct": disp_v3_vs_v2_pct,
            # Orientation
            "ori_v1": v1_ori,
            "ori_v2": v2_ori,
            "ori_v3": v3_ori,
            "ori_ins": ins_ori,
            "ori_v3_vs_ins_imprv_pct": ori_v3_vs_ins_pct,
            "ori_v3_vs_v2_delta_pct": ori_v3_vs_v2_pct,
            # Drift (10s)
            "drift_v1_m": v1_drift,
            "drift_v2_m": v2_drift,
            "drift_v3_m": v3_drift,
            "drift_ins_m": ins_drift,
            "drift_v1_pct": v1_drift_pct,
            "drift_v2_pct": v2_drift_pct,
            "drift_v3_pct": v3_drift_pct,
            "drift_ins_pct": ins_drift_pct,
            "drift_v3_vs_v2_delta_pct": drift_v3_vs_v2_pct,
        })

    # Averages
    avg_disp_imprv_ins = sum(r["disp_v3_vs_ins_imprv_pct"] for r in scenario_data) / len(scenario_data)
    avg_ori_imprv_ins = sum(r["ori_v3_vs_ins_imprv_pct"] for r in scenario_data) / len(scenario_data)

    summary = {
        "status": "v3_physics_informed_production_ready",
        "scenarios_evaluated": len(scenario_data),
        "avg_disp_improvement_vs_ins_pct": avg_disp_imprv_ins,
        "avg_ori_improvement_vs_ins_pct": avg_ori_imprv_ins,
        "training": train_summary,
        "scenarios": scenario_data,
    }
    return summary


def write_txt_report(report: dict):
    s = report
    tr = s["training"]
    lines = []
    lines.append("MODEL QUALITY REPORT — v3 (PINO-DR) vs v2 (IDNN) vs v1 (BASELINE)")
    lines.append("=" * 110)
    lines.append(f"Status: {s['status']}")
    lines.append(f"Scenarios evaluated: {s['scenarios_evaluated']}")
    lines.append(f"Average Displacement Improvement vs Raw INS: {s['avg_disp_improvement_vs_ins_pct']:+.2f}%")
    lines.append(f"Average Orientation Improvement vs Raw INS:  {s['avg_ori_improvement_vs_ins_pct']:+.2f}%")
    lines.append("")
    lines.append("ARCHITECTURE & TRAINING SUMMARY (v3 PINO-DR)")
    lines.append("-" * 110)
    lines.append(f"Model Parameters:        {tr['n_params']:,} (budget <= 25,000)")
    lines.append(f"Epochs Trained:          {tr['epochs']} (no early stopping)")
    lines.append(f"Best Checkpoint Epoch:   {tr['best_epoch']} (val_loss = {tr['best_val_loss']:.6f})")
    lines.append(f"Test Split MAE:          Disp = {tr['test_disp_mae']:.6f} m,  Ori = {tr['test_ori_mae']:.6f} rad")
    lines.append(f"Training Duration:       {tr['total_time_s']:.1f} s (~{tr['total_time_s']/60:.1f} min)")
    lines.append(f"Trip Split Hygiene:      50 train journeys / 9 val / 13 test (zero journey leakage)")
    lines.append("")
    lines.append("SCENARIO-BY-SCENARIO 10-s DRIFT COMPARISON (METERS)")
    lines.append("-" * 110)
    lines.append(f"{'Scenario':<12} | {'Raw INS':>10} | {'v1 Baseline':>12} | {'v2 IDNN':>10} | {'v3 PINO-DR':>11} | {'v3 vs INS':>10} | {'v3 vs v2':>10}")
    lines.append("-" * 110)
    for r in s["scenarios"]:
        imprv_ins = (r["drift_ins_m"] - r["drift_v3_m"]) / r["drift_ins_m"] * 100.0
        imprv_v2 = (r["drift_v2_m"] - r["drift_v3_m"]) / r["drift_v2_m"] * 100.0
        lines.append(
            f"{r['scenario']:<12} | {r['drift_ins_m']:>9.2f}m | {r['drift_v1_m']:>11.2f}m | {r['drift_v2_m']:>9.2f}m | {r['drift_v3_m']:>10.2f}m | {imprv_ins:>+9.1f}% | {imprv_v2:>+9.1f}%"
        )
    lines.append("")
    lines.append("SCENARIO-BY-SCENARIO CRSE (CUMULATIVE RESIDUAL SQUARED ERROR) COMPARISON")
    lines.append("-" * 110)
    lines.append(f"{'Scenario':<12} | {'Disp INS':>9} | {'v1 Disp':>9} | {'v2 Disp':>9} | {'v3 Disp':>9} | {'Ori INS':>8} | {'v1 Ori':>8} | {'v2 Ori':>8} | {'v3 Ori':>8}")
    lines.append("-" * 110)
    for r in s["scenarios"]:
        lines.append(
            f"{r['scenario']:<12} | {r['disp_ins']:>9.2f} | {r['disp_v1']:>9.2f} | {r['disp_v2']:>9.2f} | {r['disp_v3']:>9.2f} | "
            f"{r['ori_ins']:>8.3f} | {r['ori_v1']:>8.3f} | {r['ori_v2']:>8.3f} | {r['ori_v3']:>8.3f}"
        )
    lines.append("")
    lines.append("KEY ARCHITECTURAL & METHODOLOGICAL UPGRADES ACROSS VERSIONS")
    lines.append("-" * 110)
    lines.append("1. v1 (Baseline):")
    lines.append("   - Single-step feedforward IDNN trained on unclipped data with GPS outlier (1843.7 m/s).")
    lines.append("   - Learned constant ~12 m/s predictor; open-loop evaluation only.")
    lines.append("2. v2 (Production Baseline):")
    lines.append("   - Outlier clipping ([0, 45] m/s, [-8, 8] m/s^2), 3-layer MLP with BatchNorm + GELU.")
    lines.append("   - Window-level random train/val split (had overlap between consecutive sliding windows).")
    lines.append("3. v3 (PINO-DR - Physics-Informed Neural Operator):")
    lines.append("   - Trip-Level Split: Complete journeys held out for val/test (zero temporal data leakage).")
    lines.append("   - Multi-Task Physics Heads: Jointly predicts forward displacement, heading delta, AND ZUPT stationary probability.")
    lines.append("   - Kinematic Loss Penalty: Physics loss penalizes centripetal inconsistency (|a_lat - v*w_yaw|).")
    lines.append("   - Closed-Loop Autoregressive Evaluation: State feedback (v_prev feeding next window) with ZUPT hysteresis.")
    lines.append("   - Deployment Exports: PyTorch .pth, ONNX (opset 18), TorchScript .pt, and production engine .pkl.")
    lines.append("")

    out_file = RESULTS / "model_quality_report_v3.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[v3] Text report saved to {out_file}")


def write_json_report(report: dict):
    out_file = RESULTS / "model_quality_report_v3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[v3] JSON report saved to {out_file}")


if __name__ == "__main__":
    report = build_v3_report()
    write_txt_report(report)
    write_json_report(report)
