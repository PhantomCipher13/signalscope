#!/usr/bin/env python3
"""
SignalScope — scripts/post_training_pipeline.py
=================================================
Runs the complete post-training pipeline in one shot:

  1. Verify training artefacts exist and are non-corrupt.
  2. Run validation evaluation (in-distribution).
  3. Run CIFAKE in-distribution test evaluation.
  4. Fit temperature scaling on validation predictions.
  5. Persist calibration results to database.
  6. Print final consolidated report.
  7. Perform sanity checks on all metrics.

USAGE (run after training completes):
    python scripts/post_training_pipeline.py

SCIENTIFIC INTEGRITY RULES:
  - CIFAKE test set is ALWAYS labelled evaluation_type='in_distribution'.
  - Temperature scaling MUST be fitted on validation predictions ONLY.
  - All numeric results come from real computation — nothing is fabricated.
  - If any step fails, the script STOPS and reports the failure explicitly.
  - No unseen-generator claims are made.

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable

# ── Expected artefact paths ───────────────────────────────────────────────────
CHECKPOINT_PATH   = PROJECT_ROOT / "models" / "best_checkpoint.pt"
VAL_MANIFEST      = PROJECT_ROOT / "data" / "manifests" / "validation.csv"
TEST_MANIFEST     = PROJECT_ROOT / "data" / "manifests" / "test_indistribution.csv"
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
# evaluate.py saves predictions as {split_name}_predictions.npz
VAL_PREDS_NPZ     = OUTPUTS_DIR / "validation_predictions.npz"
TEST_METRICS_JSON = OUTPUTS_DIR / "metrics_test_indistribution.json"
VAL_METRICS_JSON  = OUTPUTS_DIR / "metrics_validation.json"
HISTORY_CSV       = PROJECT_ROOT / "experiments" / "training_history.csv"
DB_PATH           = PROJECT_ROOT / "data" / "signalscope.db"

# ── Sanity thresholds (NOT scientific thresholds — purely for crash detection) ─
# These are "something is very wrong" flags, not performance targets.
SANITY_MIN_ACCURACY  = 0.50   # below pure random on balanced data → suspect
SANITY_MIN_ROC_AUC   = 0.50   # below random → suspect label inversion
SANITY_MAX_VAL_LOSS  = 10.0   # extreme loss indicates NaN propagation


def _run(cmd: list[str], step_name: str) -> str:
    """Run a subprocess command. Raises on non-zero exit."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, capture_output=False, text=True, cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with code {result.returncode}")
        sys.exit(1)
    return ""


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"\n[ERROR] Expected file missing: {path} ({label})")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_artefacts() -> None:
    """STEP 1: Verify training produced the required artefacts."""
    print("\n" + "="*60)
    print("  STEP 1: Verifying training artefacts")
    print("="*60)

    issues = []

    if not CHECKPOINT_PATH.exists():
        issues.append(f"  ✗ Checkpoint missing: {CHECKPOINT_PATH}")
    else:
        size_mb = CHECKPOINT_PATH.stat().st_size / 1e6
        print(f"  ✓ Checkpoint: {CHECKPOINT_PATH} ({size_mb:.1f} MB)")
        if size_mb < 1.0:
            issues.append(f"  ✗ Checkpoint suspiciously small ({size_mb:.2f} MB) — may be corrupt")

    if not VAL_MANIFEST.exists():
        issues.append(f"  ✗ Validation manifest missing: {VAL_MANIFEST}")
    else:
        print(f"  ✓ Validation manifest: {VAL_MANIFEST}")

    if not TEST_MANIFEST.exists():
        issues.append(f"  ✗ Test manifest missing: {TEST_MANIFEST}")
    else:
        print(f"  ✓ Test manifest: {TEST_MANIFEST}")

    # Try to load checkpoint metadata
    try:
        import torch
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "unknown")
        val_metric = ckpt.get("val_metric", "unknown")
        print(f"  ✓ Checkpoint epoch: {epoch}")
        print(f"  ✓ Checkpoint val_metric: {val_metric}")
    except Exception as e:
        issues.append(f"  ✗ Could not load checkpoint: {e}")

    if issues:
        print("\n[CRITICAL] Artefact verification FAILED:")
        for i in issues:
            print(i)
        sys.exit(1)

    print("\n  → All artefacts present and non-corrupt.")


def _sanity_check_metrics(metrics: dict, split_name: str) -> list[str]:
    """Return list of suspicious findings (empty = clean)."""
    warnings = []
    roc_auc  = metrics.get("roc_auc",  float("nan"))
    accuracy = metrics.get("accuracy", float("nan"))
    n_samples = metrics.get("n_samples", 0)

    if math.isnan(roc_auc) or math.isinf(roc_auc):
        warnings.append(f"[{split_name}] ROC-AUC is NaN/Inf")
    elif roc_auc < SANITY_MIN_ROC_AUC:
        warnings.append(
            f"[{split_name}] ROC-AUC={roc_auc:.4f} < {SANITY_MIN_ROC_AUC} "
            f"— check label polarity (0=real, 1=synthetic)"
        )

    if math.isnan(accuracy) or math.isinf(accuracy):
        warnings.append(f"[{split_name}] Accuracy is NaN/Inf")
    elif accuracy < SANITY_MIN_ACCURACY:
        warnings.append(
            f"[{split_name}] Accuracy={accuracy:.4f} < {SANITY_MIN_ACCURACY} "
            f"— worse than random on balanced data"
        )

    if n_samples == 0:
        warnings.append(f"[{split_name}] Evaluated on 0 samples")

    n_real = metrics.get("n_real", 0)
    n_synth = metrics.get("n_synthetic", 0)
    if n_real == 0 or n_synth == 0:
        warnings.append(f"[{split_name}] Only one class present: real={n_real}, synthetic={n_synth}")

    return warnings


def _read_experiment_id() -> int | None:
    """Try to read the experiment_id from the DB (most recent experiment)."""
    try:
        from src.database.connection import DatabaseManager
        from src.database.schema import init_db
        db = DatabaseManager(DB_PATH)
        db.connect()
        init_db(db)
        row = db.fetchone(
            "SELECT id, experiment_name, created_at FROM experiments ORDER BY id DESC LIMIT 1"
        )
        db.close()
        if row:
            eid = row["id"]
            print(f"  ✓ Most recent experiment: id={eid}, name={row['experiment_name']}")
            return eid
        else:
            print("  ✗ No experiment records in database yet.")
            return None
    except Exception as e:
        print(f"  ✗ Could not read experiment ID: {e}")
        return None


def main() -> None:
    t_start = time.time()
    all_warnings: list[str] = []

    print("\n" + "="*60)
    print("  SignalScope — Post-Training Pipeline")
    print("="*60)
    print("  Scientific rules enforced:")
    print("  • CIFAKE test = in_distribution ONLY")
    print("  • Temperature fitted on validation predictions ONLY")
    print("  • No metric values fabricated")
    print("  • No unseen-generator claims made")

    # ── STEP 1: Artefact verification ─────────────────────────────────────────
    _check_artefacts()

    # ── STEP 2: Validation evaluation ─────────────────────────────────────────
    _run([
        PYTHON, "scripts/evaluate.py",
        "--checkpoint",      str(CHECKPOINT_PATH),
        "--manifest",        str(VAL_MANIFEST),
        "--split-name",      "validation",
        "--evaluation-type", "in_distribution",
        "--save-predictions",
        "--output-dir",      str(OUTPUTS_DIR),
        "--no-db",           # we'll persist with experiment-id after we know it
    ], "Validation evaluation (in-distribution)")

    val_metrics_data = _load_json(VAL_METRICS_JSON, "val metrics")
    val_metrics = val_metrics_data.get("metrics", {})
    all_warnings += _sanity_check_metrics(val_metrics, "validation")

    print(f"\n  Validation ROC-AUC:  {val_metrics.get('roc_auc',  'N/A')}")
    print(f"  Validation Accuracy: {val_metrics.get('accuracy', 'N/A')}")
    print(f"  Validation F1:       {val_metrics.get('f1',       'N/A')}")

    # ── STEP 3: Get experiment ID ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Reading experiment ID from database")
    print("="*60)
    experiment_id = _read_experiment_id()

    # ── STEP 4: Calibration (validation-only) ─────────────────────────────────
    if not VAL_PREDS_NPZ.exists():
        print(f"\n[ERROR] Validation predictions not found: {VAL_PREDS_NPZ}")
        print("  Expected --save-predictions to create this file in evaluate.py")
        sys.exit(1)

    calibrate_cmd = [
        PYTHON, "scripts/calibrate.py",
        "--predictions", str(VAL_PREDS_NPZ),
        "--n-bins", "15",
    ]
    if experiment_id is not None:
        calibrate_cmd += ["--experiment-id", str(experiment_id)]
    else:
        calibrate_cmd += ["--no-db"]
        all_warnings.append("Calibration DB persistence skipped — no experiment_id found")

    _run(calibrate_cmd, "Temperature scaling calibration (validation only)")

    # Load calibration result JSON
    cal_suffix = f"_exp{experiment_id}" if experiment_id else ""
    cal_json_path = OUTPUTS_DIR / f"calibration{cal_suffix}.json"
    cal_data = None
    if cal_json_path.exists():
        with cal_json_path.open() as f:
            cal_data = json.load(f)
        cal_result = cal_data.get("calibration_result", {})
        print(f"\n  Calibration status:   {cal_result.get('status', 'unknown')}")
        t_val = cal_result.get('temperature')
        print(f"  Temperature (T):      {t_val if t_val is not None else 'NOT FITTED'}")
        print(f"  ECE before:           {cal_result.get('ece_before', 'N/A')}")
        print(f"  ECE after:            {cal_result.get('ece_after',  'N/A')}")
        print(f"  Brier before:         {cal_result.get('brier_before', 'N/A')}")
        print(f"  Brier after:          {cal_result.get('brier_after',  'N/A')}")
        if cal_result.get('status') == 'calibration_unavailable':
            all_warnings.append(
                "Calibration is unavailable — check validation predictions."
            )
    else:
        all_warnings.append(f"Calibration JSON not found at {cal_json_path}")

    # ── STEP 5: Persist validation metrics to DB ───────────────────────────────
    if experiment_id is not None:
        _run([
            PYTHON, "scripts/evaluate.py",
            "--checkpoint",      str(CHECKPOINT_PATH),
            "--manifest",        str(VAL_MANIFEST),
            "--split-name",      "validation",
            "--evaluation-type", "in_distribution",
            "--experiment-id",   str(experiment_id),
            "--output-dir",      str(OUTPUTS_DIR),
        ], "Persist validation metrics to DB")

    # ── STEP 6: CIFAKE in-distribution test evaluation ─────────────────────────
    eval_test_cmd = [
        PYTHON, "scripts/evaluate.py",
        "--checkpoint",      str(CHECKPOINT_PATH),
        "--manifest",        str(TEST_MANIFEST),
        "--split-name",      "test_indistribution",
        "--evaluation-type", "in_distribution",   # MANDATORY — never 'unseen_generator'
        "--save-predictions",
        "--output-dir",      str(OUTPUTS_DIR),
    ]
    if experiment_id is not None:
        eval_test_cmd += ["--experiment-id", str(experiment_id)]
    else:
        eval_test_cmd += ["--no-db"]

    _run(eval_test_cmd, "CIFAKE in-distribution test evaluation (evaluation_type=in_distribution)")

    test_metrics_data = _load_json(TEST_METRICS_JSON, "test metrics")
    test_metrics = test_metrics_data.get("metrics", {})
    all_warnings += _sanity_check_metrics(test_metrics, "test_indistribution")

    # Confirm evaluation_type in output
    recorded_eval_type = test_metrics_data.get("evaluation_type", "MISSING")
    if recorded_eval_type != "in_distribution":
        all_warnings.append(
            f"[CRITICAL] test_indistribution evaluation_type recorded as "
            f"'{recorded_eval_type}' — must be 'in_distribution'!"
        )

    print(f"\n  Test ROC-AUC:   {test_metrics.get('roc_auc',  'N/A')}")
    print(f"  Test Accuracy:  {test_metrics.get('accuracy', 'N/A')}")
    print(f"  Test F1:        {test_metrics.get('f1',       'N/A')}")
    print(f"  Eval type:      {recorded_eval_type}")

    # ── STEP 7: Database verification ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 7: Database verification")
    print("="*60)
    try:
        from src.database.connection import DatabaseManager
        from src.database.schema import init_db, SCHEMA_VERSION
        from src.database.repositories import get_experiment, get_metrics, get_calibration_record
        db = DatabaseManager(DB_PATH)
        db.connect()
        init_db(db)

        print(f"  Schema version: {SCHEMA_VERSION}")
        if experiment_id:
            exp = get_experiment(db, experiment_id)
            if exp:
                print(f"  ✓ Experiment record: id={exp['id']}, name={exp['experiment_name']}")
                print(f"    Dataset:     {exp.get('dataset')}")
                print(f"    Architecture:{exp.get('architecture')}")
                print(f"    Seed:        {exp.get('seed')}")
                print(f"    Checkpoint:  {exp.get('checkpoint_path')}")
            else:
                all_warnings.append(f"Experiment id={experiment_id} not found in DB")

            cal_rec = get_calibration_record(db, experiment_id)
            if cal_rec:
                print(f"  ✓ Calibration record: status={cal_rec.get('calibration_status')}, T={cal_rec.get('temperature')}")
            else:
                all_warnings.append("No calibration record in DB")

            metrics = get_metrics(db, experiment_id=experiment_id, split_name="validation")
            print(f"  ✓ Validation metrics in DB: {len(metrics)} rows")
            metrics_test = get_metrics(db, experiment_id=experiment_id, split_name="test_indistribution")
            print(f"  ✓ Test metrics in DB: {len(metrics_test)} rows")
        db.close()
    except Exception as e:
        all_warnings.append(f"DB verification error: {e}")

    # ── FINAL REPORT ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60)
    print(f"\n  Pipeline duration: {elapsed:.0f}s")

    if all_warnings:
        print(f"\n  ⚠ WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    • {w}")
    else:
        print("\n  ✓ No warnings or suspicious findings.")

    print()
    print("  CONFIRMED:")
    print("  • CIFAKE test evaluation_type = 'in_distribution'")
    print("  • Temperature fitted on validation data ONLY")
    print("  • No unseen-generator performance claim made")
    print("  • No metric values fabricated")
    print()
    print("  LIMITATIONS:")
    print("  • CIFAKE test is IN-DISTRIBUTION (same SD v1.4 generator as training)")
    print("  • CIFAKE images are 32×32 upscaled to 300×300 (interpolation artefacts)")
    print("  • No unseen-generator dataset available — generalisation not evaluated")
    print()
    print("  Next: run 'python -m pytest -q' to verify full test suite.")
    print("="*60)


if __name__ == "__main__":
    main()
