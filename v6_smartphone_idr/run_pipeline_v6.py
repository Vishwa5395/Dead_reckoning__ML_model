"""
run_pipeline_v6.py
------------------
Master Scientific Pipeline Orchestrator for Dual-Specialist PINO-DR v6.

Adheres strictly to the 9-Stage Scientific Order:
  Step 1: Audit and Preprocess (Timestamp sync, gyro alignment, zero-centered scaling)
  Step 2 & 3: Build & Train Specialists A and B Independently
  Step 4: Evaluate Each Specialist Independently on Validation Data
  Step 5 & 6: Simple Decoupled Fusion & Closed-Loop Validation Drift Evaluation
  Step 7: Routing / Fusion Experimentation (Benchmarked strictly against Decoupled Baseline)
  Step 8: Freeze Configuration based ONLY on Validation Performance
  Step 9: Run Untouched Test Set Exactly Once for Final Reporting
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))


def run_command(cmd: list[str]):
    print(f"\n[Pipeline Runner] Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(WS_ROOT))
    if ret.returncode != 0:
        print(f"[Pipeline Runner] ERROR: Command failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="PINO-DR v6 Pipeline")
    parser.add_argument(
        "--step",
        choices=[
            "1_preprocess",
            "2_3_train",
            "4_eval_specialists",
            "5_6_eval_val_drift",
            "7_route_experiments",
            "9_final_test",
            "full_pipeline",
            "5_preprocess",
            "5_train",
            "5_eval_val",
            "5_eval_test",
            "5_full",
        ],
        default="5_eval_val",
        help="Pipeline step to execute",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--fusion", choices=["DECOUPLED", "UNCERTAINTY_BLEND", "ROUTER"], default="DECOUPLED")
    args = parser.parse_args()

    py = sys.executable

    # --- 2-Specialist Workflow ---
    if args.step == "1_preprocess" or args.step == "full_pipeline":
        print("\n>>> PIPELINE STEP 1: AUDIT & PREPROCESSING")
        run_command([py, "-u", "v6_smartphone_idr/src/preprocess_specialists.py"])

    if args.step == "2_3_train" or args.step == "full_pipeline":
        print("\n>>> PIPELINE STEP 2 & 3: INDEPENDENT SPECIALIST TRAINING")
        run_command([py, "-u", "v6_smartphone_idr/src/train_specialists.py", "--model", "BOTH", "--epochs", str(args.epochs)])

    if args.step == "4_eval_specialists" or args.step == "full_pipeline":
        print("\n>>> PIPELINE STEP 4: INDEPENDENT SPECIALIST DIAGNOSTICS")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_specialists.py", "--step", "4_diag"])

    if args.step == "5_6_eval_val_drift" or args.step == "full_pipeline":
        print(f"\n>>> PIPELINE STEP 5 & 6: CLOSED-LOOP VALIDATION DRIFT [{args.fusion}]")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_specialists.py", "--step", "6_val_drift", "--fusion", args.fusion])

    if args.step == "7_route_experiments":
        print("\n>>> PIPELINE STEP 7: ROUTING EXPERIMENTS ON VALIDATION DATA")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_specialists.py", "--step", "7_experiments"])

    if args.step == "9_final_test":
        print("\n>>> PIPELINE STEP 9: FINAL UNTOUCHED TEST BENCHMARK")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_specialists.py", "--step", "9_final_test"])

    # --- 5-Independent-Specialist Workflow ---
    if args.step == "5_preprocess" or args.step == "5_full":
        print("\n>>> 5-SPECIALIST: MULTI-SCALE PREPROCESSING")
        run_command([py, "-u", "v6_smartphone_idr/src/preprocess_five_specialists.py"])

    if args.step == "5_train" or args.step == "5_full":
        print("\n>>> 5-SPECIALIST: INDEPENDENT SPECIALIST TRAINING")
        run_command([py, "-u", "v6_smartphone_idr/src/train_five_specialists.py"])

    if args.step == "5_eval_val" or args.step == "5_full":
        print("\n>>> 5-SPECIALIST: CLOSED-LOOP VALIDATION DRIFT")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_five_specialists.py", "val"])

    if args.step == "5_eval_test" or args.step == "5_full":
        print("\n>>> 5-SPECIALIST: FINAL TEST SET BENCHMARK")
        run_command([py, "-u", "v6_smartphone_idr/src/evaluate_five_specialists.py", "test"])


if __name__ == "__main__":
    main()
