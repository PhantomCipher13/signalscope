"""
tests/unit/test_calibration_db.py
====================================
Unit tests for Phase 3 database calibration/reliability persistence.

Tests use in-memory/temporary SQLite databases — never touch production DB.
These are SOFTWARE CORRECTNESS tests only.

Tests verify:
  - insert_calibration_record with all fields
  - insert_calibration_record with NULL numeric fields (not yet fitted)
  - get_calibration_record retrieval
  - calibration_record with test split is REJECTED (scientific integrity)
  - update_analysis_reliability with partial fields
  - stability=None stored as NULL, not as 0 or 1
  - schema v2 migration creates calibration_records table
  - new analysis_runs columns exist after migration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import DatabaseManager
from src.database.schema import init_db, SCHEMA_VERSION
from src.database.repositories import (
    insert_experiment,
    insert_analysis,
    insert_calibration_record,
    get_calibration_record,
    update_analysis_reliability,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh temporary database for each test."""
    db = DatabaseManager(tmp_path / "test.db")
    db.connect()
    init_db(db)
    yield db
    db.close()


@pytest.fixture
def experiment_id(db):
    """Insert a minimal experiment record and return its ID."""
    return insert_experiment(
        db,
        experiment_name="test_exp",
        dataset="CIFAKE",
        model_name="efficientnet_b3",
        seed=42,
    )


@pytest.fixture
def analysis_id(db, experiment_id):
    """Insert a minimal analysis_run and return its ID."""
    return insert_analysis(
        db,
        image_hash="a" * 64,
        model_version="efficientnet_b3_v1",
        experiment_id=experiment_id,
        baseline_probability=0.82,
        prediction="synthetic",
        predicted_class=1,
    )


# ── 1. Schema version ─────────────────────────────────────────────────────────

class TestSchemaVersion:
    def test_schema_version_is_2(self):
        assert SCHEMA_VERSION == 2

    def test_calibration_records_table_exists(self, db):
        conn = db.connect()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_records'"
        ).fetchone()
        assert result is not None, "calibration_records table must exist after init_db"

    def test_analysis_runs_has_calibrated_probability(self, db):
        conn = db.connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()}
        assert "calibrated_probability" in cols

    def test_analysis_runs_has_reliability_status(self, db):
        conn = db.connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()}
        assert "reliability_status" in cols

    def test_analysis_runs_has_observation_count(self, db):
        conn = db.connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()}
        assert "observation_count" in cols

    def test_analysis_runs_has_std_probability(self, db):
        conn = db.connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()}
        assert "std_probability" in cols


# ── 2. insert_calibration_record ──────────────────────────────────────────────

class TestInsertCalibrationRecord:
    def test_insert_returns_id(self, db, experiment_id):
        cid = insert_calibration_record(
            db, experiment_id,
            temperature=1.25,
            calibration_status="calibrated",
            fitted_on_split="validation",
            n_samples_fitted=15000,
            ece_before=0.08,
            ece_after=0.04,
            brier_before=0.12,
            brier_after=0.09,
        )
        assert isinstance(cid, int)
        assert cid >= 1

    def test_insert_with_null_temperature_allowed(self, db, experiment_id):
        """Temperature=None must be allowed — calibration may not have been run yet."""
        cid = insert_calibration_record(
            db, experiment_id,
            temperature=None,
            calibration_status="not_calibrated",
        )
        assert cid >= 1
        record = get_calibration_record(db, experiment_id)
        assert record["temperature"] is None

    def test_insert_with_all_null_metrics_allowed(self, db, experiment_id):
        """All metric fields may be NULL until calibration actually runs."""
        cid = insert_calibration_record(
            db, experiment_id,
            temperature=None,
            calibration_status="calibration_unavailable",
            ece_before=None,
            ece_after=None,
            brier_before=None,
            brier_after=None,
        )
        record = get_calibration_record(db, experiment_id)
        assert record["ece_before"] is None
        assert record["ece_after"] is None
        assert record["brier_before"] is None
        assert record["brier_after"] is None

    def test_insert_rejects_test_split(self, db, experiment_id):
        """Scientific integrity: calibration must NEVER be fitted on test data."""
        with pytest.raises(ValueError, match="test"):
            insert_calibration_record(
                db, experiment_id,
                temperature=1.1,
                calibration_status="calibrated",
                fitted_on_split="test_indistribution",  # BLOCKED
            )

    def test_insert_rejects_test_split_any_test_name(self, db, experiment_id):
        """Any split name containing 'test' must be rejected."""
        with pytest.raises(ValueError):
            insert_calibration_record(
                db, experiment_id,
                fitted_on_split="test",
            )

    def test_insert_allows_validation_split(self, db, experiment_id):
        cid = insert_calibration_record(
            db, experiment_id,
            fitted_on_split="validation",
            calibration_status="calibrated",
            temperature=1.1,
        )
        assert cid >= 1

    def test_insert_allows_none_split(self, db, experiment_id):
        """fitted_on_split=None is allowed (calibration not yet run)."""
        cid = insert_calibration_record(
            db, experiment_id,
            fitted_on_split=None,
            calibration_status="not_calibrated",
        )
        assert cid >= 1

    def test_insert_stores_all_fields_correctly(self, db, experiment_id):
        insert_calibration_record(
            db, experiment_id,
            temperature=1.35,
            calibration_status="calibrated",
            fitted_on_split="validation",
            n_samples_fitted=15000,
            ece_before=0.082,
            ece_after=0.041,
            brier_before=0.115,
            brier_after=0.088,
            calibration_file="outputs/calibration_1.json",
            notes="Phase 3 test",
        )
        record = get_calibration_record(db, experiment_id)
        assert abs(record["temperature"] - 1.35) < 1e-6
        assert record["calibration_status"] == "calibrated"
        assert record["fitted_on_split"] == "validation"
        assert record["n_samples_fitted"] == 15000
        assert abs(record["ece_before"] - 0.082) < 1e-6
        assert abs(record["ece_after"] - 0.041) < 1e-6
        assert record["calibration_file"] == "outputs/calibration_1.json"

    def test_cascades_on_experiment_delete(self, db, experiment_id):
        insert_calibration_record(
            db, experiment_id,
            calibration_status="not_calibrated",
        )
        # Delete experiment
        db.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
        db.commit()
        record = get_calibration_record(db, experiment_id)
        assert record is None  # should be cascade-deleted


# ── 3. get_calibration_record ─────────────────────────────────────────────────

class TestGetCalibrationRecord:
    def test_returns_none_when_no_record(self, db, experiment_id):
        record = get_calibration_record(db, experiment_id)
        assert record is None

    def test_returns_most_recent_when_multiple(self, db, experiment_id):
        insert_calibration_record(
            db, experiment_id, calibration_status="not_calibrated", temperature=None
        )
        insert_calibration_record(
            db, experiment_id, calibration_status="calibrated", temperature=1.4
        )
        record = get_calibration_record(db, experiment_id)
        assert record["calibration_status"] == "calibrated"  # most recent

    def test_returns_dict(self, db, experiment_id):
        insert_calibration_record(db, experiment_id, calibration_status="not_calibrated")
        record = get_calibration_record(db, experiment_id)
        assert isinstance(record, dict)


# ── 4. update_analysis_reliability ───────────────────────────────────────────

class TestUpdateAnalysisReliability:
    def test_update_sets_reliability_status(self, db, analysis_id):
        update_analysis_reliability(
            db, analysis_id,
            reliability_status="insufficient_evidence",
        )
        row = db.fetchone("SELECT reliability_status FROM analysis_runs WHERE id=?", (analysis_id,))
        assert row["reliability_status"] == "insufficient_evidence"

    def test_update_sets_calibrated_probability(self, db, analysis_id):
        update_analysis_reliability(
            db, analysis_id,
            calibrated_probability=0.77,
        )
        row = db.fetchone("SELECT calibrated_probability FROM analysis_runs WHERE id=?", (analysis_id,))
        assert abs(row["calibrated_probability"] - 0.77) < 1e-6

    def test_update_stability_none_stored_as_null(self, db, analysis_id):
        """stability=None (insufficient evidence) must be stored as NULL, not 0 or 1."""
        update_analysis_reliability(
            db, analysis_id,
            reliability_status="insufficient_evidence",
            # stability NOT passed = remains NULL
        )
        row = db.fetchone("SELECT stability FROM analysis_runs WHERE id=?", (analysis_id,))
        assert row["stability"] is None

    def test_update_observation_count(self, db, analysis_id):
        update_analysis_reliability(
            db, analysis_id,
            observation_count=5,
            std_probability=0.035,
            stability=0.965,
        )
        row = db.fetchone(
            "SELECT observation_count, std_probability, stability FROM analysis_runs WHERE id=?",
            (analysis_id,),
        )
        assert row["observation_count"] == 5
        assert abs(row["std_probability"] - 0.035) < 1e-6
        assert abs(row["stability"] - 0.965) < 1e-6

    def test_update_with_no_fields_is_noop(self, db, analysis_id):
        """Calling update with no fields should not raise and not change anything."""
        update_analysis_reliability(db, analysis_id)
        # Should complete without error

    def test_update_calibration_status(self, db, analysis_id):
        update_analysis_reliability(
            db, analysis_id,
            calibration_status="calibrated",
            calibrated_probability=0.68,
        )
        row = db.fetchone(
            "SELECT calibration_status, calibrated_probability FROM analysis_runs WHERE id=?",
            (analysis_id,),
        )
        assert row["calibration_status"] == "calibrated"
        assert abs(row["calibrated_probability"] - 0.68) < 1e-6

    def test_update_full_reliability_result(self, db, analysis_id):
        """Simulate storing a complete reliability engine output."""
        update_analysis_reliability(
            db, analysis_id,
            calibrated_probability=0.81,
            calibration_status="calibrated",
            reliability_status="stable",
            reliability_note="Stability computed from 5 observations.",
            observation_count=5,
            mean_probability=0.85,
            std_probability=0.012,
            stability=0.988,
        )
        row = db.fetchone("SELECT * FROM analysis_runs WHERE id=?", (analysis_id,))
        assert row["reliability_status"] == "stable"
        assert row["observation_count"] == 5
        assert abs(row["stability"] - 0.988) < 1e-6


# ── 5. API Schema validation ──────────────────────────────────────────────────

class TestAPISchema:
    def test_analysis_response_schema_importable(self):
        from src.schemas import AnalysisResponse, ReliabilitySchema, CalibrationStatusEnum
        assert AnalysisResponse is not None

    def test_reliability_schema_stability_none_allowed(self):
        from src.schemas import ReliabilitySchema, StabilityStatusEnum, ReliabilityStatusEnum
        schema = ReliabilitySchema(
            observation_count=1,
            mean_probability=0.82,
            standard_deviation=None,
            stability=None,
            stability_status=StabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            status=ReliabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            note="Only one observation.",
            threshold_configured=False,
        )
        assert schema.stability is None
        assert schema.standard_deviation is None

    def test_analysis_response_calibration_status_explicit(self):
        from src.schemas import (
            AnalysisResponse, ReliabilitySchema, CalibrationStatusEnum,
            StabilityStatusEnum, ReliabilityStatusEnum,
        )
        reliability = ReliabilitySchema(
            observation_count=1,
            mean_probability=0.82,
            standard_deviation=None,
            stability=None,
            stability_status=StabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            status=ReliabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            note="Single observation.",
            threshold_configured=False,
        )
        response = AnalysisResponse(
            prediction="synthetic",
            label="Likely AI-Generated",
            baseline_probability=0.82,
            calibrated_probability=None,
            calibration_status=CalibrationStatusEnum.NOT_CALIBRATED,
            reliability=reliability,
        )
        assert response.calibrated_probability is None
        assert response.calibration_status == CalibrationStatusEnum.NOT_CALIBRATED.value

    def test_analysis_response_json_serializable(self):
        import json
        from src.schemas import (
            AnalysisResponse, ReliabilitySchema, CalibrationStatusEnum,
            StabilityStatusEnum, ReliabilityStatusEnum,
        )
        reliability = ReliabilitySchema(
            observation_count=1,
            mean_probability=0.82,
            standard_deviation=None,
            stability=None,
            stability_status=StabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            status=ReliabilityStatusEnum.INSUFFICIENT_EVIDENCE,
            note="Single observation.",
            threshold_configured=False,
        )
        response = AnalysisResponse(
            prediction="synthetic",
            label="Likely AI-Generated",
            baseline_probability=0.82,
            calibrated_probability=None,
            calibration_status=CalibrationStatusEnum.NOT_CALIBRATED,
            reliability=reliability,
        )
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        assert data["calibrated_probability"] is None
        assert data["reliability"]["stability"] is None
        assert data["calibration_status"] == "not_calibrated"
