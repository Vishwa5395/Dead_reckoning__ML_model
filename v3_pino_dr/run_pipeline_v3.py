"""
run_pipeline_v3.py
------------------
Single-command driver for the PINO-DR v3 dead-reckoning pipeline:

  1. Preprocess: Extract enriched features, trip-level split, window, scale.
  2. Train: Full-epoch training with per-epoch checkpoints.
  3. Evaluate: Closed-loop autoregressive benchmark with ZUPT.

Usage:
  python run_pipeline_v3.py               # run all steps
  python run_pipeline_v3.py --step preprocess
  python run_pipeline_v3.py --step train
  python run_pipeline_v3.py --step eval
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[0]


def run_preprocess():
    from src.preprocess_v3 import main as preprocess_main
    preprocess_main()


def run_train():
    from src.train_v3 import main as train_main
    train_main()


def run_eval():
    from src.evaluate_v3 import evaluate_all
    evaluate_all()


def run_report():
    from src.report_v3 import build_v3_report, write_txt_report, write_json_report
    rep = build_v3_report()
    write_txt_report(rep)
    write_json_report(rep)


def main():
    parser = argparse.ArgumentParser(description="PINO-DR v3 Pipeline")
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "eval", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("preprocess", "all"):
        print("\n" + "=" * 80)
        print("STEP 1/3: PREPROCESSING (ENRICHED FEATURES + TRIP-LEVEL SPLIT)")
        print("=" * 80)
        run_preprocess()

    if args.step in ("train", "all"):
        print("\n" + "=" * 80)
        print("STEP 2/3: TRAINING PINO-DR v3 (FULL EPOCHS, NO EARLY STOPPING)")
        print("=" * 80)
        run_train()

    if args.step in ("eval", "all"):
        print("\n" + "=" * 80)
        print("STEP 3/3: CLOSED-LOOP BENCHMARK EVALUATION")
        print("=" * 80)
        run_eval()
        run_report()

    print("\n[v3] Pipeline complete.")


if __name__ == "__main__":
    main()
