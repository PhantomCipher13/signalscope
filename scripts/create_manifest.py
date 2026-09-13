#!/usr/bin/env python3
"""
SignalScope — scripts/create_manifest.py
=========================================
Generate a deterministic train/val/test manifest CSV from a dataset directory.

Usage:
    python scripts/create_manifest.py --root <dataset_dir>
    python scripts/create_manifest.py --root <dataset_dir> \\
        --output data/splits/manifest.csv \\
        --train 0.70 --val 0.15 --test 0.15

The manifest records for each sample:
    sample_id, path, label, label_name, class_dir, generator, split

Output is deterministic for a given seed and dataset contents.

IMPORTANT:
    Do NOT use the test split for checkpoint selection, calibration,
    or threshold tuning. The test split is for final evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_manifest
from src.utils import get_config, get_logger, setup_logging

logger = get_logger("create_manifest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a SignalScope dataset manifest."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset root directory.")
    parser.add_argument("--output", type=Path,
                        default=Path("data/splits/manifest.csv"),
                        help="Output manifest CSV path.")
    parser.add_argument("--train", type=float, default=0.70,
                        help="Training fraction.")
    parser.add_argument("--val", type=float, default=0.15,
                        help="Validation fraction.")
    parser.add_argument("--test", type=float, default=0.15,
                        help="Test fraction.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config).")
    parser.add_argument("--no-validate-images", action="store_true",
                        help="Skip image loading validation (faster).")
    args = parser.parse_args()

    cfg = get_config()
    seed = args.seed if args.seed is not None else cfg.get("seed", {}).get("value", 42)

    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=PROJECT_ROOT / cfg.get("logging", {}).get("log_dir", "logs"),
        filename=cfg.get("logging", {}).get("filename", "signalscope.log"),
    )

    frac_sum = args.train + args.val + args.test
    if abs(frac_sum - 1.0) > 1e-6:
        print(f"[ERROR] Fractions must sum to 1.0, got {frac_sum:.4f}")
        sys.exit(1)

    print(f"\nCreating manifest:")
    print(f"  Dataset root: {args.root.resolve()}")
    print(f"  Output:       {args.output}")
    print(f"  Split:        train={args.train} val={args.val} test={args.test}")
    print(f"  Seed:         {seed}")
    print()

    summary = create_manifest(
        root_dir=args.root,
        output_path=args.output,
        seed=seed,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        skip_invalid=not args.no_validate_images,
    )

    print("\n--- Manifest Summary ---")
    for k, v in summary.items():
        print(f"  {k:<25s} {v}")

    if summary.get("total", 0) == 0:
        print("\n[BLOCKED] No samples written. Cannot proceed to training.")
        print("  1. Place your dataset under the --root directory.")
        print("  2. Ensure directories are named with real/fake keywords")
        print("     (see dataset.real_dir_keywords in default.yaml).")
        print("  3. Re-run this script.")
        sys.exit(1)

    if summary.get("label_unknown_count", 0) > 0:
        print(
            f"\n[WARNING] {summary['label_unknown_count']} images have unknown labels."
            f"\n  These are excluded from the manifest."
            f"\n  Update default.yaml: dataset.real_dir_keywords / synthetic_dir_keywords"
        )

    print(f"\n[OK] Manifest written to {args.output}")
    print("     Review before training. Check 'generator' column for leakage risks.")


if __name__ == "__main__":
    main()
