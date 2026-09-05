"""
run_pipeline.py
---------------
Driver CLI script to orchestrate the AI Dead Reckoning pipeline:
1. Preprocessing IO-VNBD (S- inputs, V- labels, leveling, NHC, scaling)
2. Model Training (Displacement IDNN 40 epochs, Orientation IDNN 60 epochs, batch_size=256, Adamax, MAE)
3. Model Evaluation (Autoregressive closed-loop inference, benchmark metrics, trajectory plots)
"""

import sys
import argparse

from src.dataset_loader import prepare_and_cache_dataset
from src.train import train_all
from src.evaluate import evaluate_all_scenarios


def main():
    parser = argparse.ArgumentParser(description="IO-VNBD IDNN Dead Reckoning Pipeline")
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    args = parser.parse_args()

    if args.step in ["preprocess", "all"]:
        print("\n=======================================================")
        print("STEP 1: PREPROCESSING IO-VNBD DATASET")
        print("=======================================================")
        prepare_and_cache_dataset()

    if args.step in ["train", "all"]:
        print("\n=======================================================")
        print("STEP 2: TRAINING IDNN DISPLACEMENT & ORIENTATION MODELS")
        print("=======================================================")
        train_all()

    if args.step in ["evaluate", "all"]:
        print("\n=======================================================")
        print("STEP 3: BENCHMARK EVALUATION (AUTOREGRESSIVE INFERENCE)")
        print("=======================================================")
        evaluate_all_scenarios()


if __name__ == "__main__":
    main()
