"""
src/database/schema.py
=======================
SignalScope database schema definition and migration helpers.

Tables:
  - schema_version       : tracks migrations (for future upgrades)
  - experiments          : training run configuration + paths
  - experiment_metrics   : per-split metrics for each experiment
  - analysis_runs        : per-image inference results
  - probe_results        : reliability probe outcomes per analysis
  - provenance_records   : C2PA/EXIF metadata per analysis

Image binary data is NEVER stored.
Only hashes, paths, and structured metadata/results are persisted.

Scientific integrity notes:
  - `calibration_status` defaults to 'not_calibrated' until Phase 2
  - `uncertainty_status` defaults to 'not_computed' until reliability engine
  - `evaluation_type` in experiment_metrics distinguishes in_distribution from unseen_generator
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.database.connection import DatabaseManager

logger = logging.getLogger("signalscope.database")

# Bump this when schema changes require migration
SCHEMA_VERSION = 2

# ── DDL statements ─────────────────────────────────────────────────────────────

_DDL = """
-- Schema version tracking (for future migrations)
CREATE TABLE IF NOT EXISTS schema_version (
    version         INTEGER PRIMARY KEY,
    applied_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Training experiments: config, paths, reproducibility info
-- Model weights are NOT stored here — only checkpoint_path references them.
CREATE TABLE IF NOT EXISTS experiments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name     TEXT    NOT NULL,
    dataset             TEXT    NOT NULL,           -- e.g. 'CIFAKE'
    model_name          TEXT    NOT NULL,           -- e.g. 'efficientnet_b3'
    model_version       TEXT,
    architecture        TEXT,
    seed                INTEGER NOT NULL,
    config_path         TEXT,
    checkpoint_path     TEXT,
    train_manifest      TEXT,                       -- path to train.csv
    val_manifest        TEXT,                       -- path to validation.csv
    test_manifest       TEXT,                       -- path to test_indistribution.csv
    epochs_trained      INTEGER,
    best_epoch          INTEGER,
    batch_size          INTEGER,
    learning_rate       REAL,
    weight_decay        REAL,
    scheduler           TEXT,
    augmentation_config TEXT,                       -- JSON or YAML string (not binary)
    training_duration_s REAL,
    generator           TEXT,                       -- e.g. 'stable_diffusion_v1.4'
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Per-split metrics for each experiment
-- evaluation_type distinguishes 'in_distribution' from 'unseen_generator'
-- DO NOT record CIFAKE test as 'unseen_generator'
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    split_name      TEXT    NOT NULL,               -- 'train', 'validation', 'test_indistribution'
    evaluation_type TEXT    NOT NULL DEFAULT 'in_distribution',  -- 'in_distribution' | 'unseen_generator'
    metric_name     TEXT    NOT NULL,               -- 'roc_auc', 'accuracy', 'f1', ...
    metric_value    REAL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, split_name, metric_name)
);

-- Per-image inference/analysis results
-- image_data (binary) is NEVER stored — only image_hash and optional path
CREATE TABLE IF NOT EXISTS analysis_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash          TEXT    NOT NULL,           -- SHA-256 of input image
    image_path          TEXT,                       -- optional local path (NOT binary)
    model_version       TEXT,
    experiment_id       INTEGER REFERENCES experiments(id),
    baseline_probability REAL,                      -- raw model softmax probability
    prediction          TEXT,                       -- 'real' | 'synthetic'
    predicted_class     INTEGER,                    -- 0=real, 1=synthetic
    confidence          REAL,                       -- same as probability (Phase 1: uncalibrated)
    calibration_status  TEXT    NOT NULL DEFAULT 'not_calibrated',
    uncertainty_status  TEXT    NOT NULL DEFAULT 'not_computed',
    stability           REAL,                       -- NULL until reliability engine (Phase 2)
    probes_run          INTEGER DEFAULT 0,
    stop_reason         TEXT,                       -- NULL until adaptive probing (Phase 2)
    processing_time_ms  REAL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Reliability probe results (per transformation per analysis)
-- Populated when reliability engine is active (Phase 2+)
CREATE TABLE IF NOT EXISTS probe_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id             INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    transformation_name     TEXT    NOT NULL,
    transformation_params   TEXT,                   -- JSON string of parameters
    probability             REAL,
    success                 INTEGER NOT NULL DEFAULT 1, -- 0=failure, 1=success
    error_message           TEXT,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Provenance records (C2PA/EXIF metadata per analysis)
-- Populated when provenance module is active (Phase 2+)
CREATE TABLE IF NOT EXISTS provenance_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id         INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    metadata_available  INTEGER NOT NULL DEFAULT 0, -- 0=no, 1=yes
    metadata_summary    TEXT,
    c2pa_status         TEXT    NOT NULL DEFAULT 'not_checked',
    exif_status         TEXT    NOT NULL DEFAULT 'not_checked',
    exif_fields         TEXT,                       -- JSON string of relevant EXIF fields
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_analysis_image_hash    ON analysis_runs(image_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_created_at    ON analysis_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_exp ON experiment_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_probe_results_analysis ON probe_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_provenance_analysis    ON provenance_records(analysis_id);

-- ── Phase 3 (v2): Calibration records ────────────────────────────────────────
-- Stores calibration metadata per experiment.
-- ECE and Brier are measured results, not performance targets.
-- Values are NULL until calibration is actually performed.
-- Temperature is NEVER fabricated — NULL means not yet fitted.
CREATE TABLE IF NOT EXISTS calibration_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id        INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    temperature          REAL,                       -- NULL until fitted on validation data
    calibration_status   TEXT NOT NULL DEFAULT 'not_calibrated',
    fitted_on_split      TEXT,                       -- 'validation' or NULL (must be validation only)
    n_samples_fitted     INTEGER,                    -- number of val samples used for fitting
    ece_before           REAL,                       -- ECE before calibration (NULL if not computed)
    ece_after            REAL,                       -- ECE after calibration (NULL if not computed)
    brier_before         REAL,                       -- Brier before calibration
    brier_after          REAL,                       -- Brier after calibration
    calibration_file     TEXT,                       -- path to saved calibration JSON
    notes                TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_calibration_exp ON calibration_records(experiment_id);
"""

# ── v2 migration: adds calibration_records + new columns to analysis_runs ─────
# Safe to run on an existing v1 database — uses IF NOT EXISTS / ADD COLUMN.
_DDL_V2_MIGRATION = """
CREATE TABLE IF NOT EXISTS calibration_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id        INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    temperature          REAL,
    calibration_status   TEXT NOT NULL DEFAULT 'not_calibrated',
    fitted_on_split      TEXT,
    n_samples_fitted     INTEGER,
    ece_before           REAL,
    ece_after            REAL,
    brier_before         REAL,
    brier_after          REAL,
    calibration_file     TEXT,
    notes                TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_calibration_exp ON calibration_records(experiment_id);
"""

# Columns added to analysis_runs in v2 (ALTER TABLE if missing)
_V2_ANALYSIS_COLUMNS = [
    ("calibrated_probability", "REAL", None),
    ("reliability_status",     "TEXT", None),
    ("reliability_note",       "TEXT", None),
    ("observation_count",      "INTEGER", "0"),
    ("mean_probability",       "REAL", None),
    ("std_probability",        "REAL", None),
]


def init_db(db: "DatabaseManager") -> None:
    """
    Create all tables and insert the current schema version if not present.
    Safe to call multiple times (idempotent — uses IF NOT EXISTS).
    Runs incremental migrations for databases created at older schema versions.
    """
    conn = db.connect()
    conn.executescript(_DDL)
    conn.commit()

    # Check current recorded version
    existing = db.fetchone("SELECT MAX(version) AS v FROM schema_version")
    current_version = existing["v"] if existing and existing["v"] else 0

    # Run v2 migration if needed (adds calibration_records + new analysis columns)
    if current_version < 2:
        logger.info("Running schema v2 migration...")
        conn.executescript(_DDL_V2_MIGRATION)
        conn.commit()
        # Add new columns to analysis_runs if they don't exist (ALTER TABLE)
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()}
        for col_name, col_type, col_default in _V2_ANALYSIS_COLUMNS:
            if col_name not in existing_cols:
                default_clause = f" DEFAULT {col_default}" if col_default is not None else ""
                conn.execute(f"ALTER TABLE analysis_runs ADD COLUMN {col_name} {col_type}{default_clause}")
                logger.info("Added column analysis_runs.%s", col_name)
        conn.commit()
        db.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", (2,))
        db.commit()
        logger.info("Schema v2 migration complete.")

    # Record current schema version
    if current_version < SCHEMA_VERSION:
        db.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        db.commit()
        logger.info(f"Database schema v{SCHEMA_VERSION} initialised at {db.db_path}")
    else:
        logger.debug(f"Database schema v{SCHEMA_VERSION} already current at {db.db_path}")


def drop_all(db: "DatabaseManager") -> None:
    """
    Drop all tables (test cleanup only — NEVER call in production).
    Requires explicit confirmation to prevent accidental data loss.
    """
    tables = [
        "calibration_records", "provenance_records", "probe_results",
        "analysis_runs", "experiment_metrics", "experiments", "schema_version",
    ]
    conn = db.connect()
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    logger.warning("All database tables dropped (test/dev use only)")
