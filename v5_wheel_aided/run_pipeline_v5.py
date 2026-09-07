"""
run_pipeline_v5.py
------------------
Single-command pipeline runner for PINO-DR v5 (Wheel-Aided).

Usage:
  python run_pipeline_v5.py --step preprocess
  python run_pipeline_v5.py --step train --tag v5_step1_raw_wheel
  python run_pipeline_v5.py --step eval --ckpt best_model_v5_step1_raw_wheel.pth
  python run_pipeline_v5.py --step report
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
    from v5_wheel_aided.src.preprocess_v5 import main as prep_main
    prep_main()


def run_train(tag="v5_step1_raw_wheel"):
    from v5_wheel_aided.src.train_v5 import run_training
    run_training(tag=tag)


def run_eval(ckpt="best_model_v5_step1_raw_wheel.pth", suffix="_step1"):
    from v5_wheel_aided.src.evaluate_v5 import evaluate_all
    evaluate_all(ckpt_name=ckpt, out_suffix=suffix)


def main():
    parser = argparse.ArgumentParser(description="PINO-DR v5 Pipeline")
    parser.add_argument("--step", choices=["preprocess", "train", "eval", "all"], default="eval",
                        help="Pipeline step to run")
    parser.add_argument("--tag", default="v5_step1_raw_wheel", help="Experiment tag")
    parser.add_argument("--ckpt", default="best_model_v5_step1_raw_wheel.pth", help="Checkpoint filename")
    parser.add_argument("--suffix", default="_step1", help="Output suffix")
    args = parser.parse_args()

    if args.step in ["preprocess", "all"]:
        run_preprocess()
    if args.step in ["train", "all"]:
        run_train(tag=args.tag)
    if args.step in ["eval", "all"]:
        run_eval(ckpt=args.ckpt, suffix=args.suffix)


if __name__ == "__main__":
    main()
