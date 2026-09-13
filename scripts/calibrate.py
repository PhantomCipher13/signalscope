#!/usr/bin/env python3
"""
SignalScope — scripts/calibrate.py
====================================
Post-training temperature scaling calibration.

This script fits a temperature scalar T on the VALIDATION set predictions
saved during training (outputs/val_predictions.npz), then persists the
calibration record to the database.

SCIENTIFIC INTEGRITY RULES:
  - Temperature MUST be fitted on validation data ONLY.
  - The test split MUST NOT be used for calibration.
  - The unseen-generator split MUST NOT be used for calibration.
  - ECE and Brier scores are measured results, not performance targets.
  - If fitting fails or insufficient data exists, calibration_status
    is explicitly set to 'calibration_unavailable'.

Usage:
    python scripts/calibrate.py --predictions outputs/val_predictions.npz \\
                                 --experiment-id 1

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils import get_config, get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit temperature scaling on validation predictions.")
    p.add_argument("--predictions", type=Path, default=Path("outputs/val_predictions.npz"),
                   help="Path to .npz file with 'labels' and 'probs' arrays (from training).")
    p.add_argument("--experiment-id", type=int, default=None,
                   help="Database experiment ID to associate calibration with.")
    p.add_argument("--save-scaler", type=Path, default=None,
                   help="Path to save fitted calibration JSON. "
                        "Defaults to outputs/calibration_<experiment_id>.json")
    p.add_argument("--n-bins", type=int, default=15,
                   help="Number of bins for ECE calculation.")
    p.add_argument("--no-db", action="store_true",
                   help="Skip database persistence.")
    p.add_argument("--device", type=str, default=None, help="Unused; for API consistency.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_dir=PROJECT_ROOT / log_cfg.get("log_dir", "logs"),
        filename=log_cfg.get("filename", "signalscope.log"),
    )
    logger = get_logger("calibrate")

    print("\n=== SignalScope — Temperature Scaling Calibration ===")
    print("  Fitting temperature on VALIDATION predictions only.")
    print("  Test split will NOT be used for calibration.\n")

    # ── Load predictions ──────────────────────────────────────────────────
    pred_path = PROJECT_ROOT / args.predictions
    if not pred_path.exists():
        print(f"[ERROR] Predictions file not found: {pred_path}")
        print("  Run training first to generate outputs/val_predictions.npz")
        sys.exit(1)

    data = np.load(pred_path, allow_pickle=True)
    labels = data["labels"].astype(np.int32)
    probs = data["probs"].astype(np.float32)
    split = str(data.get("split", ["validation"])[0])

    print(f"  Loaded {len(labels)} samples from: {pred_path}")
    print(f"  Split label in file: '{split}'")

    # Enforce validation-only rule
    if "test" in split.lower():
        print(f"\n[BLOCKED] Cannot calibrate on test split (split='{split}').")
        print("  Temperature must only be fitted on validation data.")
        sys.exit(1)

    if len(labels) < 10:
        print(f"\n[WARNING] Only {len(labels)} samples — calibration may be unreliable.")

    # ── Fit temperature scaler ─────────────────────────────────────────────
    from src.calibration import TemperatureScaler

    scaler = TemperatureScaler()
    result = scaler.fit(probs, labels, split="validation")

    print(f"\n  Calibration status: {result.status.value}")
    if result.temperature is not None:
        print(f"  Temperature (T):    {result.temperature:.6f}")
    else:
        print(f"  Temperature (T):    not fitted (unavailable)")

    if result.ece_before is not None:
        print(f"\n  ECE before:         {result.ece_before:.6f}")
    if result.ece_after is not None:
        print(f"  ECE after:          {result.ece_after:.6f}")
    if result.brier_before is not None:
        print(f"  Brier before:       {result.brier_before:.6f}")
    if result.brier_after is not None:
        print(f"  Brier after:        {result.brier_after:.6f}")

    print()
    if result.ece_after is not None and result.ece_before is not None:
        delta_ece = result.ece_after - result.ece_before
        print(f"  ECE change:         {delta_ece:+.6f} ({'improved' if delta_ece < 0 else 'worsened'})")

    print()
    print("  REMINDER: These are measured results, NOT performance targets.")
    print("  ECE and Brier thresholds must be validated experimentally.")

    # ── Save scaler JSON ───────────────────────────────────────────────────
    save_path = args.save_scaler
    if save_path is None:
        suffix = f"_exp{args.experiment_id}" if args.experiment_id else ""
        save_path = PROJECT_ROOT / f"outputs/calibration{suffix}.json"

    save_path = PROJECT_ROOT / save_path if not save_path.is_absolute() else save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(str(save_path))
    print(f"\n  Calibration saved to: {save_path}")

    # ── Database persistence ───────────────────────────────────────────────
    if not args.no_db:
        if args.experiment_id is None:
            print("\n[DB] No --experiment-id specified — calibration not persisted to database.")
            print("     Re-run with --experiment-id <id> to save calibration record.")
        else:
            try:
                from src.database.connection import DatabaseManager, _resolve_db_path
                from src.database.schema import init_db
                from src.database.repositories import insert_calibration_record

                db_path = _resolve_db_path()
                db = DatabaseManager(db_path)
                db.connect()
                init_db(db)

                cid = insert_calibration_record(
                    db,
                    experiment_id=args.experiment_id,
                    temperature=result.temperature,
                    calibration_status=result.status.value,
                    fitted_on_split="validation",
                    n_samples_fitted=result.n_samples_fitted,
                    ece_before=result.ece_before,
                    ece_after=result.ece_after,
                    brier_before=result.brier_before,
                    brier_after=result.brier_after,
                    calibration_file=str(save_path),
                    notes=(
                        "Temperature scaling fitted on validation predictions. "
                        "ECE and Brier are measured values, not targets."
                    ),
                )
                db.close()
                print(f"\n  [DB] Calibration persisted: calibration_record id={cid}")
            except Exception as db_err:
                print(f"\n  [DB WARNING] Could not persist calibration: {db_err}")

    print("\n=== Calibration complete ===")
    print("  You can now use the calibrated scaler for inference.")
    print("  DO NOT re-fit on the test split.")


if __name__ == "__main__":
    main()
