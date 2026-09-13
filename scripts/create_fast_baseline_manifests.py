#!/usr/bin/env python3
"""
scripts/create_fast_baseline_manifests.py
==========================================
Creates a reproducible balanced subset of CIFAKE for rapid development.

Fast baseline spec:
  - 10,000 training images  (5,000 real + 5,000 synthetic) — from train.csv
  - 3,000 validation images (1,500 real + 1,500 synthetic) — from validation.csv
  - 20,000 test images — full test_indistribution.csv unchanged

All splits maintain generator identity (CIFAKE = SD v1.4 throughout).
Subset is sampled with fixed seed for reproducibility.
Does NOT modify original manifests.

Outputs:
  data/manifests/fast_train.csv
  data/manifests/fast_validation.csv
  (test_indistribution.csv reused as-is — no subsetting needed)

SCIENTIFIC NOTE:
  These manifests are for development/rapid baseline only.
  The fast baseline CANNOT claim better generalisation than the full run.
  Results must be labelled 'fast_baseline_dev' in the database.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED             = 42
TRAIN_N_PER_CLASS = 5_000   # 10k total
VAL_N_PER_CLASS   = 1_500   # 3k total

SRC_TRAIN = PROJECT_ROOT / "data/manifests/train.csv"
SRC_VAL   = PROJECT_ROOT / "data/manifests/validation.csv"
OUT_TRAIN = PROJECT_ROOT / "data/manifests/fast_train.csv"
OUT_VAL   = PROJECT_ROOT / "data/manifests/fast_validation.csv"


def balanced_sample(rows: list[dict], n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    real  = [r for r in rows if str(r.get("label", "")) == "0"]
    synth = [r for r in rows if str(r.get("label", "")) == "1"]
    real_s  = rng.sample(real,  min(n_per_class, len(real)))
    synth_s = rng.sample(synth, min(n_per_class, len(synth)))
    combined = real_s + synth_s
    rng.shuffle(combined)
    return combined


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("\n=== Creating Fast Baseline Manifests ===")
    print(f"  Seed: {SEED}")
    print(f"  Train target: {TRAIN_N_PER_CLASS * 2:,} images ({TRAIN_N_PER_CLASS:,}/class)")
    print(f"  Val target:   {VAL_N_PER_CLASS   * 2:,} images ({VAL_N_PER_CLASS:,}/class)")

    # Load source manifests
    with SRC_TRAIN.open(encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with SRC_VAL.open(encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    print(f"\n  Source train:  {len(train_rows):,} rows")
    print(f"  Source val:    {len(val_rows):,} rows")

    # Sample
    fast_train = balanced_sample(train_rows, TRAIN_N_PER_CLASS, seed=SEED)
    fast_val   = balanced_sample(val_rows,   VAL_N_PER_CLASS,   seed=SEED + 1)

    # Verify balance
    ft_real  = sum(1 for r in fast_train if str(r.get("label","")) == "0")
    ft_synth = sum(1 for r in fast_train if str(r.get("label","")) == "1")
    fv_real  = sum(1 for r in fast_val   if str(r.get("label","")) == "0")
    fv_synth = sum(1 for r in fast_val   if str(r.get("label","")) == "1")

    print(f"\n  Fast train:    {len(fast_train):,} (real={ft_real}, synth={ft_synth})")
    print(f"  Fast val:      {len(fast_val):,} (real={fv_real}, synth={fv_synth})")

    if ft_real != ft_synth or fv_real != fv_synth:
        print("  [WARNING] Class imbalance detected — check source manifests.")

    # Write
    write_csv(fast_train, OUT_TRAIN)
    write_csv(fast_val,   OUT_VAL)

    print(f"\n  Written: {OUT_TRAIN}")
    print(f"  Written: {OUT_VAL}")
    print(f"\n  Test manifest: data/manifests/test_indistribution.csv (full, unchanged)")
    print("\n  These manifests are for FAST DEVELOPMENT BASELINE only.")
    print("  Results must be labelled as 'fast_baseline_dev' — NOT final results.")
    print("=== Done ===")


if __name__ == "__main__":
    main()
