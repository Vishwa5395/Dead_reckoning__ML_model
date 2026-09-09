"""
run_pipeline_v7.py
------------------
Single-command pipeline runner for PINO-DR v7 (Dual-Specialist Fine-Tuned Model).

Usage:
  python run_pipeline_v7.py                  # runs full pipeline (split, train, eval, report)
  python run_pipeline_v7.py --step split
  python run_pipeline_v7.py --step train
  python run_pipeline_v7.py --step eval
  python run_pipeline_v7.py --step report
  python run_pipeline_v7.py --eval-untuned   # benchmark zero-retraining baseline
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
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))


def run_split():
    from v7_sept_model.src.split_data_v7 import split_data
    split_data()


def run_train(epochs=40, patience=10):
    from v7_sept_model.src.train_v7 import train_both
    train_both(max_epochs=epochs, patience=patience)


def run_eval(use_untuned=False, tag="v7_production", mode="production", s1_source="tuned"):
    from v7_sept_model.src.evaluate_v7 import evaluate_all
    evaluate_all(use_untuned=use_untuned, tag=tag, measure_mode=mode, s1_source=s1_source)


def run_report():
    from v7_sept_model.src.report_v7 import generate_report
    generate_report()


def main():
    parser = argparse.ArgumentParser(description="PINO-DR v7 Pipeline Driver")
    parser.add_argument(
        "--step",
        choices=["split", "train", "eval", "report", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs per specialist")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--eval-untuned", action="store_true", help="Also evaluate untuned baseline switch")
    args = parser.parse_args()

    if args.step in ("split", "all"):
        print("\n" + "=" * 90)
        print("STEP 1: DATA SPLIT (60TH PERCENTILE YAW RATE POOLING WITH ANCHOR REPLAY)")
        print("=" * 90)
        run_split()

    if args.step in ("train", "all"):
        print("\n" + "=" * 90)
        print("STEP 2: FINE-TUNE SPECIALISTS (ANCHOR REPLAY POOLS WITH CONSERVATIVE LR)")
        print("=" * 90)
        run_train(epochs=args.epochs, patience=args.patience)

    if args.eval_untuned:
        print("\n" + "=" * 90)
        print("STEP 3A: EVALUATE UNTUNED BASELINE SWITCH (v3 + v4-D ZERO-RETRAINING)")
        print("=" * 90)
        run_eval(use_untuned=True, tag="v7_untuned", mode="standard")

    if args.step in ("eval", "all"):
        print("\n" + "=" * 90)
        print("STEP 3B: EVALUATE PRODUCTION DUAL-SPECIALIST FUSION (LEARNED YAW GATING)")
        print("=" * 90)
        run_eval(use_untuned=False, tag="v7_production", mode="production", s1_source="tuned")
        run_eval(use_untuned=False, tag="v7_highway_anchor", mode="production", s1_source="v3")

    if args.step in ("report", "all"):
        print("\n" + "=" * 90)
        print("STEP 4: GENERATE COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 90)
        run_report()

    print("\n[v7] Pipeline complete.")


if __name__ == "__main__":
    main()
