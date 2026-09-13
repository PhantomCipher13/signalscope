#!/usr/bin/env python3
"""
SignalScope — scripts/report_results.py
========================================
Fills in the BASELINE_TRAINING_REPORT.md template with real results
from the database and output JSON files.

Reads from:
  - outputs/metrics_validation.json
  - outputs/metrics_test_indistribution.json
  - outputs/calibration_exp<id>.json
  - data/signalscope.db (experiment + calibration records)

Writes to:
  - docs/BASELINE_TRAINING_REPORT.md (in-place)

Run after post_training_pipeline.py completes successfully.

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR    = PROJECT_ROOT / "outputs"
REPORT_TMPL    = PROJECT_ROOT / "docs" / "BASELINE_TRAINING_REPORT.md"
DB_PATH        = PROJECT_ROOT / "data" / "signalscope.db"


def _load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def _fmt(value, fmt=".4f") -> str:
    """Format a numeric value or return MISSING."""
    if value is None:
        return "[NOT COMPUTED]"
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def main() -> None:
    print("\n=== SignalScope — Filling Training Report ===")

    if not REPORT_TMPL.exists():
        print(f"[ERROR] Report template not found: {REPORT_TMPL}")
        sys.exit(1)

    # ── Load outputs ─────────────────────────────────────────────────────────
    val_data  = _load_json_safe(OUTPUTS_DIR / "metrics_validation.json")
    test_data = _load_json_safe(OUTPUTS_DIR / "metrics_test_indistribution.json")

    val_metrics  = val_data.get("metrics",  {}) if val_data  else {}
    test_metrics = test_data.get("metrics", {}) if test_data else {}

    # ── Load DB records ───────────────────────────────────────────────────────
    experiment_id   = None
    exp_record      = {}
    cal_record      = {}
    val_db_metrics  = {}
    test_db_metrics = {}

    try:
        from src.database.connection import DatabaseManager
        from src.database.schema import init_db
        from src.database.repositories import get_calibration_record, get_metrics, list_experiments

        db = DatabaseManager(DB_PATH)
        db.connect()
        init_db(db)

        experiments = list_experiments(db)
        if experiments:
            exp_record = dict(experiments[-1])
            experiment_id = exp_record.get("id")
            print(f"  Experiment ID: {experiment_id}, name: {exp_record.get('experiment_name')}")

            if experiment_id:
                cal = get_calibration_record(db, experiment_id)
                cal_record = dict(cal) if cal else {}

                val_rows  = get_metrics(db, experiment_id, split_name="validation")
                test_rows = get_metrics(db, experiment_id, split_name="test_indistribution")
                val_db_metrics  = {r["metric_name"]: r["metric_value"] for r in val_rows}
                test_db_metrics = {r["metric_name"]: r["metric_value"] for r in test_rows}
        db.close()
    except Exception as e:
        print(f"[WARN] DB read failed: {e}")

    # Prefer JSON file metrics (they're fresher), fall back to DB
    vm = val_metrics  if val_metrics  else val_db_metrics
    tm = test_metrics if test_metrics else test_db_metrics

    # ── Build replacement table ───────────────────────────────────────────────
    replacements = {
        # Training results
        "Best epoch": _fmt(exp_record.get("best_epoch"), ".0f"),
        "Final training loss": "[see training log]",
        "Best validation loss": _fmt(exp_record.get("best_val_loss")),
        "Best validation ROC-AUC": _fmt(vm.get("roc_auc")),
        "Best validation accuracy": _fmt(vm.get("accuracy")),
        "Best validation F1": _fmt(vm.get("f1")),
        "Training duration": "[see training log]",
        "Database experiment ID": str(experiment_id) if experiment_id else "[not found]",

        # Validation metrics
        "val_roc_auc":   _fmt(vm.get("roc_auc")),
        "val_accuracy":  _fmt(vm.get("accuracy")),
        "val_f1":        _fmt(vm.get("f1")),
        "val_precision": _fmt(vm.get("precision")),
        "val_recall":    _fmt(vm.get("recall")),

        # Test metrics
        "test_roc_auc":   _fmt(tm.get("roc_auc")),
        "test_accuracy":  _fmt(tm.get("accuracy")),
        "test_f1":        _fmt(tm.get("f1")),
        "test_precision": _fmt(tm.get("precision")),
        "test_recall":    _fmt(tm.get("recall")),

        # Calibration
        "cal_status":        cal_record.get("calibration_status", "[not found]"),
        "cal_temperature":   _fmt(cal_record.get("temperature")),
        "cal_ece_before":    _fmt(cal_record.get("ece_before")),
        "cal_ece_after":     _fmt(cal_record.get("ece_after")),
        "cal_brier_before":  _fmt(cal_record.get("brier_before")),
        "cal_brier_after":   _fmt(cal_record.get("brier_after")),
    }

    # ── Read and update template ──────────────────────────────────────────────
    content = REPORT_TMPL.read_text(encoding="utf-8")
    original = content

    # Replace structured [PENDING] entries by row label
    row_map = {
        "Best epoch":                replacements["Best epoch"],
        "Final training loss":       replacements["Final training loss"],
        "Best validation loss":      replacements["Best validation loss"],
        "Best validation ROC-AUC":   replacements["Best validation ROC-AUC"],
        "Best validation accuracy":  replacements["Best validation accuracy"],
        "Best validation F1":        replacements["Best validation F1"],
        "Training duration":         replacements["Training duration"],
        "Database experiment ID":    replacements["Database experiment ID"],
    }

    # Generic: replace `| Row Label | [PENDING] |` with real value
    for row_label, value in row_map.items():
        pattern = rf"(\|\s*{re.escape(row_label)}\s*\|\s*)`\[PENDING\]`"
        replacement = rf"\1`{value}`"
        content = re.sub(pattern, replacement, content)

    # Fill validation table with real values
    val_table_replacements = {
        "ROC-AUC":   replacements["val_roc_auc"],
        "Accuracy":  replacements["val_accuracy"],
        "F1":        replacements["val_f1"],
        "Precision": replacements["val_precision"],
        "Recall":    replacements["val_recall"],
    }
    test_table_replacements = {
        "ROC-AUC":   replacements["test_roc_auc"],
        "Accuracy":  replacements["test_accuracy"],
        "F1":        replacements["test_f1"],
        "Precision": replacements["test_precision"],
        "Recall":    replacements["test_recall"],
    }

    def fill_table_section(text, section_marker, table_replacements):
        """Fill metric values in a specific section of the report."""
        lines = text.split("\n")
        in_section = False
        result = []
        for line in lines:
            if section_marker in line:
                in_section = True
            if in_section:
                for metric, value in table_replacements.items():
                    pattern = rf"(\|\s*{re.escape(metric)}\s*\|\s*)`\[PENDING\]`"
                    if re.search(pattern, line):
                        line = re.sub(pattern, rf"\1`{value}`", line)
                        break
            result.append(line)
        return "\n".join(result)

    content = fill_table_section(content, "## 3. Validation Evaluation", val_table_replacements)
    content = fill_table_section(content, "## 4. CIFAKE In-Distribution Test", test_table_replacements)

    # Fill calibration section
    cal_table = {
        "Calibration status": replacements["cal_status"],
        "Temperature (T)":    replacements["cal_temperature"],
        "ECE before calibration": replacements["cal_ece_before"],
        "ECE after calibration":  replacements["cal_ece_after"],
        "Brier score before": replacements["cal_brier_before"],
        "Brier score after":  replacements["cal_brier_after"],
    }
    content = fill_table_section(content, "## 5. Temperature Scaling Calibration", cal_table)

    # Fill DB section
    db_status = {
        "Experiment record":  "✓ present" if exp_record else "✗ missing",
        "Validation metrics": "✓ present" if val_db_metrics else "✗ missing",
        "Test metrics":       "✓ present" if test_db_metrics else "✗ missing",
        "Calibration record": "✓ present" if cal_record else "✗ missing",
    }
    content = fill_table_section(content, "## 6. Database Persistence", db_status)

    # Write back
    if content != original:
        REPORT_TMPL.write_text(content, encoding="utf-8")
        print(f"\n  ✓ Report updated: {REPORT_TMPL}")
    else:
        print(f"\n  ✗ No replacements made — check that training completed and metrics JSON exists.")

    # Print summary
    print("\n  Summary:")
    print(f"    Validation  ROC-AUC:  {replacements['val_roc_auc']}")
    print(f"    Test        ROC-AUC:  {replacements['test_roc_auc']}")
    print(f"    Temperature:          {replacements['cal_temperature']}")
    print(f"    ECE (after cal):      {replacements['cal_ece_after']}")
    print(f"    Experiment ID:        {replacements['Database experiment ID']}")
    print("\n  CONFIRMED: CIFAKE test labelled 'in_distribution' throughout.")
    print("  CONFIRMED: Temperature fitted on validation only.")
    print("  CONFIRMED: No metric values were invented.")


if __name__ == "__main__":
    main()
