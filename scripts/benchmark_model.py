#!/usr/bin/env python3
"""Run a benchmark pass for the trained dead-reckoning IDNN models.

This script resolves the project root automatically so it works from this
workspace without hardcoded Windows paths. It can optionally preprocess the
source data first, then evaluates all benchmark scenarios and saves a JSON
summary plus the usual plot files under results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import prepare_and_cache_dataset
from src.evaluate import evaluate_all_scenarios


def ensure_dataset(cache_dir: Path) -> None:
    required = [
        cache_dir / "dataset_splits.npz",
        cache_dir / "scalers.pkl",
        cache_dir / "test_scenarios.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset artifacts are missing. Run preprocessing first. Missing: "
            + ", ".join(missing)
        )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the trained displacement/orientation IDNN models."
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip dataset preprocessing if the cached .npz/.pkl files already exist.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild the cached preprocessed dataset before evaluating.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "preprocessed",
        help="Directory containing dataset splits and scaler files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="Directory containing the trained .pth model checkpoint files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory where benchmark plots and summary files are written.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    results_dir = args.results_dir.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.force_preprocess or not args.skip_preprocess:
        if args.force_preprocess or not (
            (cache_dir / "dataset_splits.npz").exists()
            and (cache_dir / "scalers.pkl").exists()
            and (cache_dir / "test_scenarios.pkl").exists()
        ):
            print("[Benchmark] Preprocessing dataset...")
            prepare_and_cache_dataset(cache_dir=str(cache_dir))
    else:
        ensure_dataset(cache_dir)

    print(f"[Benchmark] Loading checkpoints from: {checkpoint_dir}")
    print(f"[Benchmark] Evaluating and saving results to: {results_dir}")

    summary = evaluate_all_scenarios(
        cache_dir=str(cache_dir),
        checkpoint_dir=str(checkpoint_dir),
        results_dir=str(results_dir),
    )

    json_path = results_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print(f"\n[Benchmark] Done. JSON summary saved to: {json_path}")
    print(f"[Benchmark] Plots and pickle summary are in: {results_dir}")


if __name__ == "__main__":
    main()
