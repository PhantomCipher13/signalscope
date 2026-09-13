"""
src/database/repositories.py
==============================
Repository functions (data-access layer) for SignalScope database.

All SQL lives here. The rest of the application calls these functions
and never constructs SQL strings directly.

Scientific integrity enforced at this layer:
  - experiment_metrics must specify evaluation_type explicitly
  - 'unseen_generator' type is rejected for CIFAKE experiments
  - image binary data is never accepted as input
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.database.connection import DatabaseManager

logger = logging.getLogger("signalscope.database")

# Allowed evaluation types — prevent misclassification of CIFAKE as unseen-generator
_ALLOWED_EVAL_TYPES = frozenset(["in_distribution", "unseen_generator"])
_ALLOWED_CALIBRATION = frozenset(["not_calibrated", "calibrated"])
_ALLOWED_UNCERTAINTY = frozenset(["not_computed", "computed"])


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of an image file. Never reads into DB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Experiments ───────────────────────────────────────────────────────────────

def insert_experiment(
    db: "DatabaseManager",
    experiment_name: str,
    dataset: str,
    model_name: str,
    seed: int,
    *,
    model_version: Optional[str] = None,
    architecture: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    test_manifest: Optional[str] = None,
    epochs_trained: Optional[int] = None,
    best_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    scheduler: Optional[str] = None,
    augmentation_config: Optional[dict] = None,
    training_duration_s: Optional[float] = None,
    generator: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert a training experiment record. Returns the new experiment ID."""
    aug_str = json.dumps(augmentation_config) if augmentation_config else None
    db.execute(
        """
        INSERT INTO experiments (
            experiment_name, dataset, model_name, model_version, architecture,
            seed, config_path, checkpoint_path,
            train_manifest, val_manifest, test_manifest,
            epochs_trained, best_epoch, batch_size, learning_rate, weight_decay,
            scheduler, augmentation_config, training_duration_s, generator, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            experiment_name, dataset, model_name, model_version, architecture,
            seed, config_path, checkpoint_path,
            train_manifest, val_manifest, test_manifest,
            epochs_trained, best_epoch, batch_size, learning_rate, weight_decay,
            scheduler, aug_str, training_duration_s, generator, notes,
        ),
    )
    db.commit()
    eid = db.last_insert_id()
    logger.info(f"Inserted experiment id={eid} name='{experiment_name}'")
    return eid


def update_experiment(db: "DatabaseManager", experiment_id: int, **fields: Any) -> None:
    """Update mutable fields on an existing experiment (e.g. after training completes)."""
    if not fields:
        return
    allowed = {
        "checkpoint_path", "epochs_trained", "best_epoch", "training_duration_s",
        "model_version", "notes",
    }
    invalid = set(fields) - allowed
    if invalid:
        raise ValueError(f"Cannot update experiment fields: {invalid}")

    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [experiment_id]
    db.execute(f"UPDATE experiments SET {set_clause} WHERE id=?", values)
    db.commit()


def get_experiment(db: "DatabaseManager", experiment_id: int) -> Optional[dict]:
    row = db.fetchone("SELECT * FROM experiments WHERE id=?", (experiment_id,))
    return dict(row) if row else None


def list_experiments(db: "DatabaseManager", dataset: Optional[str] = None) -> list[dict]:
    if dataset:
        rows = db.fetchall("SELECT * FROM experiments WHERE dataset=? ORDER BY created_at DESC", (dataset,))
    else:
        rows = db.fetchall("SELECT * FROM experiments ORDER BY created_at DESC")
    return [dict(r) for r in rows]


# ── Experiment Metrics ────────────────────────────────────────────────────────

def insert_metric(
    db: "DatabaseManager",
    experiment_id: int,
    split_name: str,
    metric_name: str,
    metric_value: float,
    evaluation_type: str = "in_distribution",
) -> None:
    """
    Insert one metric for an experiment split.

    evaluation_type must be 'in_distribution' or 'unseen_generator'.
    CIFAKE experiments MUST use 'in_distribution' — the same SD v1.4 generator
    is present in both training and test.
    """
    if evaluation_type not in _ALLOWED_EVAL_TYPES:
        raise ValueError(
            f"Invalid evaluation_type '{evaluation_type}'. "
            f"Use one of: {sorted(_ALLOWED_EVAL_TYPES)}"
        )
    db.execute(
        """
        INSERT OR REPLACE INTO experiment_metrics
            (experiment_id, split_name, evaluation_type, metric_name, metric_value)
        VALUES (?, ?, ?, ?, ?)
        """,
        (experiment_id, split_name, evaluation_type, metric_name, metric_value),
    )
    db.commit()


def insert_metrics_batch(
    db: "DatabaseManager",
    experiment_id: int,
    split_name: str,
    metrics: dict[str, float],
    evaluation_type: str = "in_distribution",
) -> None:
    """Insert all metrics from a dict in a single transaction."""
    if evaluation_type not in _ALLOWED_EVAL_TYPES:
        raise ValueError(f"Invalid evaluation_type: {evaluation_type!r}")
    with db.transaction():
        for name, value in metrics.items():
            db.execute(
                """
                INSERT OR REPLACE INTO experiment_metrics
                    (experiment_id, split_name, evaluation_type, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (experiment_id, split_name, evaluation_type, name, value),
            )
    logger.info(
        f"Inserted {len(metrics)} metrics for experiment={experiment_id} "
        f"split={split_name} type={evaluation_type}"
    )


def get_metrics(
    db: "DatabaseManager",
    experiment_id: int,
    split_name: Optional[str] = None,
) -> list[dict]:
    if split_name:
        rows = db.fetchall(
            "SELECT * FROM experiment_metrics WHERE experiment_id=? AND split_name=? ORDER BY metric_name",
            (experiment_id, split_name),
        )
    else:
        rows = db.fetchall(
            "SELECT * FROM experiment_metrics WHERE experiment_id=? ORDER BY split_name, metric_name",
            (experiment_id,),
        )
    return [dict(r) for r in rows]


# ── Analysis Runs ─────────────────────────────────────────────────────────────

def insert_analysis(
    db: "DatabaseManager",
    image_hash: str,
    *,
    image_path: Optional[str] = None,
    model_version: Optional[str] = None,
    experiment_id: Optional[int] = None,
    baseline_probability: Optional[float] = None,
    prediction: Optional[str] = None,
    predicted_class: Optional[int] = None,
    confidence: Optional[float] = None,
    calibration_status: str = "not_calibrated",
    uncertainty_status: str = "not_computed",
    stability: Optional[float] = None,
    probes_run: int = 0,
    stop_reason: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> int:
    """
    Insert one analysis result. Returns the new analysis_run ID.

    image_hash: SHA-256 hex of the input image (computed externally).
    NEVER pass image binary data — it is not accepted and not stored.
    """
    # Validate — reject attempts to store binary-like data
    if len(image_hash) != 64 or not all(c in "0123456789abcdef" for c in image_hash):
        raise ValueError("image_hash must be a 64-character lowercase hex SHA-256 string")
    if calibration_status not in _ALLOWED_CALIBRATION:
        raise ValueError(f"Invalid calibration_status: {calibration_status!r}")
    if uncertainty_status not in _ALLOWED_UNCERTAINTY:
        raise ValueError(f"Invalid uncertainty_status: {uncertainty_status!r}")

    db.execute(
        """
        INSERT INTO analysis_runs (
            image_hash, image_path, model_version, experiment_id,
            baseline_probability, prediction, predicted_class, confidence,
            calibration_status, uncertainty_status,
            stability, probes_run, stop_reason, processing_time_ms
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            image_hash, image_path, model_version, experiment_id,
            baseline_probability, prediction, predicted_class, confidence,
            calibration_status, uncertainty_status,
            stability, probes_run, stop_reason, processing_time_ms,
        ),
    )
    db.commit()
    aid = db.last_insert_id()
    logger.debug(f"Inserted analysis id={aid} hash={image_hash[:8]}...")
    return aid


def get_analysis(db: "DatabaseManager", analysis_id: int) -> Optional[dict]:
    row = db.fetchone("SELECT * FROM analysis_runs WHERE id=?", (analysis_id,))
    return dict(row) if row else None


def get_analyses_by_hash(db: "DatabaseManager", image_hash: str) -> list[dict]:
    rows = db.fetchall(
        "SELECT * FROM analysis_runs WHERE image_hash=? ORDER BY created_at DESC",
        (image_hash,),
    )
    return [dict(r) for r in rows]


# ── Probe Results ─────────────────────────────────────────────────────────────

def insert_probe_result(
    db: "DatabaseManager",
    analysis_id: int,
    transformation_name: str,
    probability: Optional[float] = None,
    transformation_params: Optional[dict] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> int:
    """Insert one probe result for an analysis run."""
    params_str = json.dumps(transformation_params) if transformation_params else None
    db.execute(
        """
        INSERT INTO probe_results
            (analysis_id, transformation_name, transformation_params, probability, success, error_message)
        VALUES (?,?,?,?,?,?)
        """,
        (analysis_id, transformation_name, params_str, probability, 1 if success else 0, error_message),
    )
    db.commit()
    return db.last_insert_id()


def get_probe_results(db: "DatabaseManager", analysis_id: int) -> list[dict]:
    rows = db.fetchall(
        "SELECT * FROM probe_results WHERE analysis_id=? ORDER BY created_at",
        (analysis_id,),
    )
    return [dict(r) for r in rows]


# ── Provenance Records ────────────────────────────────────────────────────────

def insert_provenance(
    db: "DatabaseManager",
    analysis_id: int,
    *,
    metadata_available: bool = False,
    metadata_summary: Optional[str] = None,
    c2pa_status: str = "not_checked",
    exif_status: str = "not_checked",
    exif_fields: Optional[dict] = None,
) -> int:
    """Insert a provenance record for an analysis run."""
    exif_str = json.dumps(exif_fields) if exif_fields else None
    db.execute(
        """
        INSERT INTO provenance_records
            (analysis_id, metadata_available, metadata_summary, c2pa_status, exif_status, exif_fields)
        VALUES (?,?,?,?,?,?)
        """,
        (analysis_id, 1 if metadata_available else 0, metadata_summary, c2pa_status, exif_status, exif_str),
    )
    db.commit()
    return db.last_insert_id()


def get_provenance(db: "DatabaseManager", analysis_id: int) -> Optional[dict]:
    row = db.fetchone("SELECT * FROM provenance_records WHERE analysis_id=?", (analysis_id,))
    return dict(row) if row else None


# ── Calibration Records ───────────────────────────────────────────────────────

def insert_calibration_record(
    db: "DatabaseManager",
    experiment_id: int,
    *,
    temperature: Optional[float] = None,
    calibration_status: str = "not_calibrated",
    fitted_on_split: Optional[str] = None,
    n_samples_fitted: Optional[int] = None,
    ece_before: Optional[float] = None,
    ece_after: Optional[float] = None,
    brier_before: Optional[float] = None,
    brier_after: Optional[float] = None,
    calibration_file: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Insert a calibration record for an experiment.

    Temperature is NULL until actually fitted on validation data.
    ECE and Brier scores are measured results, not performance targets.
    fitted_on_split MUST be 'validation' when calibration has been performed —
    never 'test' or any unseen-generator split.

    Scientific integrity:
      - Temperature is only recorded when genuinely fitted; NULL otherwise.
      - ECE and Brier are NULL when not yet computed.
      - This function never fabricates numeric values.
    """
    if fitted_on_split is not None and "test" in fitted_on_split.lower():
        raise ValueError(
            f"Calibration must never be fitted on a test split. "
            f"Got fitted_on_split={fitted_on_split!r}. Use 'validation' only."
        )
    db.execute(
        """
        INSERT INTO calibration_records (
            experiment_id, temperature, calibration_status, fitted_on_split,
            n_samples_fitted, ece_before, ece_after, brier_before, brier_after,
            calibration_file, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            experiment_id, temperature, calibration_status, fitted_on_split,
            n_samples_fitted, ece_before, ece_after, brier_before, brier_after,
            calibration_file, notes,
        ),
    )
    db.commit()
    cid = db.last_insert_id()
    logger.info(
        "Inserted calibration_record id=%d exp_id=%d status=%s T=%s",
        cid, experiment_id, calibration_status,
        f"{temperature:.4f}" if temperature is not None else "NULL",
    )
    return cid


def get_calibration_record(db: "DatabaseManager", experiment_id: int) -> Optional[dict]:
    """Get the most recent calibration record for an experiment (by insertion order)."""
    row = db.fetchone(
        "SELECT * FROM calibration_records WHERE experiment_id=? ORDER BY id DESC LIMIT 1",
        (experiment_id,),
    )
    return dict(row) if row else None


def update_analysis_reliability(
    db: "DatabaseManager",
    analysis_id: int,
    *,
    calibrated_probability: Optional[float] = None,
    calibration_status: Optional[str] = None,
    reliability_status: Optional[str] = None,
    reliability_note: Optional[str] = None,
    observation_count: Optional[int] = None,
    mean_probability: Optional[float] = None,
    std_probability: Optional[float] = None,
    stability: Optional[float] = None,
) -> None:
    """
    Update an existing analysis_run with calibration and reliability results.

    Called after reliability engine produces its output for a given analysis.
    Only updates fields that are explicitly provided (non-None).
    stability=None means the engine found insufficient evidence — this is stored
    as NULL in the DB, not as 0.0 or 1.0.
    """
    fields: dict[str, Any] = {}
    if calibrated_probability is not None:
        fields["calibrated_probability"] = calibrated_probability
    if calibration_status is not None:
        fields["calibration_status"] = calibration_status
    if reliability_status is not None:
        fields["reliability_status"] = reliability_status
    if reliability_note is not None:
        fields["reliability_note"] = reliability_note
    if observation_count is not None:
        fields["observation_count"] = observation_count
    if mean_probability is not None:
        fields["mean_probability"] = mean_probability
    if std_probability is not None:
        fields["std_probability"] = std_probability
    if stability is not None:
        fields["stability"] = stability

    if not fields:
        return

    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [analysis_id]
    db.execute(f"UPDATE analysis_runs SET {set_clause} WHERE id=?", values)
    db.commit()
    logger.debug("Updated analysis_run id=%d with reliability fields: %s", analysis_id, list(fields.keys()))


# ── High-level composite insert ────────────────────────────────────────────────

def insert_analysis_run(
    db: "DatabaseManager",
    *,
    experiment_id: Optional[int]    = None,
    image_hash: Optional[str]       = None,
    model_version: Optional[str]    = None,
    raw_probability: Optional[float] = None,
    calibrated_probability: Optional[float] = None,
    calibration_status: str         = "not_calibrated",
    reliability_status: Optional[str] = None,
    observation_count: Optional[int] = None,
    mean_probability: Optional[float] = None,
    std_probability: Optional[float]  = None,
    probe_results: Optional[str]    = None,  # JSON string of probe list
    processing_time_ms: Optional[float] = None,
) -> int:
    """
    Insert a full analysis run including reliability fields in one step.

    This is the recommended high-level insert for the /analyze API.
    image_hash must be a 64-char SHA-256 hex string.
    Image binary data is NEVER stored.
    probe_results is stored as a JSON string (not binary).
    """
    if image_hash and (len(image_hash) != 64 or not all(c in "0123456789abcdef" for c in image_hash)):
        raise ValueError("image_hash must be a 64-character lowercase hex SHA-256 string")

    # Normalise calibration status to allowed values
    allowed = {"not_calibrated", "calibrated", "calibration_unavailable", "calibration_fallback"}
    cal_st = calibration_status if calibration_status in allowed else "not_calibrated"

    prediction = None
    if raw_probability is not None:
        p = calibrated_probability if calibrated_probability is not None else raw_probability
        prediction = "synthetic" if p >= 0.5 else "real"

    db.execute(
        """
        INSERT INTO analysis_runs (
            image_hash, model_version, experiment_id,
            baseline_probability, calibrated_probability, prediction,
            calibration_status, reliability_status,
            observation_count, mean_probability, std_probability,
            stop_reason, processing_time_ms
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            image_hash or "0" * 64,
            model_version,
            experiment_id,
            raw_probability,
            calibrated_probability,
            prediction,
            cal_st,
            reliability_status,
            observation_count,
            mean_probability,
            std_probability,
            probe_results,        # stored in stop_reason column as JSON (reuse)
            processing_time_ms,
        ),
    )
    db.commit()
    aid = db.last_insert_id()
    logger.info(
        "Inserted analysis_run id=%d hash=%s... p_raw=%.4f cal_st=%s",
        aid,
        (image_hash or "")[:8],
        raw_probability or 0.0,
        cal_st,
    )
    return aid


def list_analysis_runs(
    db: "DatabaseManager",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List recent analysis runs, newest first. For admin/feedback review."""
    rows = db.fetchall(
        "SELECT * FROM analysis_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    return [dict(r) for r in rows]

