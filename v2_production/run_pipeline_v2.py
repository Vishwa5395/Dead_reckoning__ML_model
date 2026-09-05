"""
run_pipeline_v2.py
------------------
One-command driver for the production-grade v2 IO-VNBD dead-reckoning pipeline:

  1. Clean the cached splits (clip GPS outliers, re-fit scalers).
  2. Train the v2 Displacement + Orientation IDNN models.
  3. Evaluate closed-loop autoregressive benchmark.
  4. Write the comparison report (v2 vs previous model).

Usage:
  python run_pipeline_v2.py               # run all steps
  python run_pipeline_v2.py --step clean  # only data cleaning
  python run_pipeline_v2.py --step train  # only training
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]


def run_clean() -> None:
    from src.preprocess_v2 import main as clean_main
    clean_main()


def run_train() -> None:
    from src.train_v2 import main as train_main
    train_main()


def run_eval() -> None:
    from src.evaluate_v2 import evaluate_all
    evaluate_all()


def run_report() -> None:
    from src.report_v2 import main as report_main
    report_main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["clean", "train", "eval", "report", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("clean", "all"):
        print("\n" + "=" * 80)
        print("STEP 1/4: CLEANING CACHED SPLITS (OUTLIER REMOVAL + RE-SCALE)")
        print("=" * 80)
        run_clean()

    if args.step in ("train", "all"):
        print("\n" + "=" * 80)
        print("STEP 2/4: TRAINING v2 IDNN MODELS")
        print("=" * 80)
        run_train()

    if args.step in ("eval", "all"):
        print("\n" + "=" * 80)
        print("STEP 3/4: CLOSED-LOOP BENCHMARK EVALUATION")
        print("=" * 80)
        run_eval()

    if args.step in ("report", "all"):
        print("\n" + "=" * 80)
        print("STEP 4/4: WRITING COMPARISON REPORT")
        print("=" * 80)
        run_report()

    print("\n[v2] Pipeline complete.")


if __name__ == "__main__":
    main()
