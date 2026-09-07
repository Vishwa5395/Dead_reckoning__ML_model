"""
run_pipeline_v4.py
------------------
Single-command pipeline runner for PINO-DR v4 (Turn-Focused).

Usage:
  python run_pipeline_v4.py               # run all steps (preprocess, train, eval, report)
  python run_pipeline_v4.py --step preprocess
  python run_pipeline_v4.py --step train
  python run_pipeline_v4.py --step eval
  python run_pipeline_v4.py --step report
  python run_pipeline_v4.py --ablation [A|B|C|D|E]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_preprocess():
    from src.preprocess_v4 import main as prep_main
    prep_main()


def run_train(ablation="E"):
    from src.train_v4 import run_training
    if ablation == "E":
        run_training(tag="v4_full")
    elif ablation == "A":
        run_training({
            "in_channels": 4, "use_multihead_attention": False,
            "use_cross_task_coupling": False, "use_turn_adaptive_loss": False,
            "use_physics_loss": False, "alpha_base": 1.2
        }, tag="ablation_A_v3_baseline")
    elif ablation == "B":
        run_training({
            "in_channels": 5, "use_multihead_attention": False,
            "use_cross_task_coupling": False, "use_turn_adaptive_loss": False,
            "use_physics_loss": False, "alpha_base": 1.2
        }, tag="ablation_B_yaw_accel")
    elif ablation == "C":
        run_training({
            "in_channels": 6, "use_multihead_attention": False,
            "use_cross_task_coupling": False, "use_turn_adaptive_loss": False,
            "use_physics_loss": False, "alpha_base": 1.2
        }, tag="ablation_C_centripetal_res")
    elif ablation == "D":
        run_training({
            "in_channels": 6, "use_multihead_attention": True,
            "use_cross_task_coupling": True, "use_turn_adaptive_loss": False,
            "use_physics_loss": False, "alpha_base": 1.2
        }, tag="ablation_D_attn_coupling")


def run_eval():
    from src.evaluate_v4 import evaluate_all
    evaluate_all()


def run_report():
    from src.report_v4 import generate_report
    generate_report()


def main():
    parser = argparse.ArgumentParser(description="PINO-DR v4 Pipeline Driver")
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "eval", "report", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    parser.add_argument(
        "--ablation",
        choices=["A", "B", "C", "D", "E"],
        default="E",
        help="Ablation model to run (default: E = Full V4)",
    )
    args = parser.parse_args()

    if args.step in ("preprocess", "all"):
        print("\n" + "=" * 80)
        print("STEP 1: PREPROCESSING (6-CHANNELS, YAW ACCEL, CENTRIPETAL RESIDUAL)")
        print("=" * 80)
        run_preprocess()

    if args.step in ("train", "all"):
        print("\n" + "=" * 80)
        print(f"STEP 2: TRAINING PINO-DR v4 (ABLATION {args.ablation}, EARLY STOPPING)")
        print("=" * 80)
        run_train(args.ablation)

    if args.step in ("eval", "all"):
        print("\n" + "=" * 80)
        print("STEP 3: CLOSED-LOOP BENCHMARK EVALUATION WITH TURN PARTITIONS")
        print("=" * 80)
        run_eval()

    if args.step in ("report", "all"):
        print("\n" + "=" * 80)
        print("STEP 4: GENERATE 5-WAY COMPARISON REPORT")
        print("=" * 80)
        run_report()

    print("\n[v4] Pipeline complete.")


if __name__ == "__main__":
    main()
