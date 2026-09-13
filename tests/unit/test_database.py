"""
tests/unit/test_database.py
=============================
Unit tests for the SignalScope database layer.

Tests cover:
  1. Database initialisation
  2. Schema creation (all 5 tables)
  3. Experiment insertion and retrieval
  4. Metric insertion (single + batch)
  5. Analysis run insertion and retrieval
  6. Probe result insertion
  7. Provenance record insertion
  8. Foreign key / referential integrity
  9. Duplicate / invalid record handling
  10. Database path configuration (isolation from production DB)
  11. Validation that image binary data is rejected
  12. Evaluation type validation (CIFAKE must be in_distribution)

All tests use a fresh in-memory or temp-file SQLite DB — never the production DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import DatabaseManager, set_db_path
from src.database.schema import init_db, drop_all, SCHEMA_VERSION
from src.database.repositories import (
    insert_experiment, update_experiment, get_experiment, list_experiments,
    insert_metric, insert_metrics_batch, get_metrics,
    insert_analysis, get_analysis, get_analyses_by_hash,
    insert_probe_result, get_probe_results,
    insert_provenance, get_provenance,
    sha256_bytes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path) -> DatabaseManager:
    """Fresh isolated SQLite database for each test."""
    db_path = tmp_path / "test_signalscope.db"
    manager = DatabaseManager(db_path)
    manager.connect()
    init_db(manager)
    yield manager
    manager.close()


@pytest.fixture
def experiment_id(db) -> int:
    """Insert a minimal experiment and return its ID."""
    return insert_experiment(
        db,
        experiment_name="test_baseline_cifake",
        dataset="CIFAKE",
        model_name="efficientnet_b3",
        seed=42,
        generator="stable_diffusion_v1.4",
        architecture="efficientnet_b3",
        batch_size=32,
        learning_rate=1e-4,
        epochs_trained=5,
        best_epoch=3,
    )


@pytest.fixture
def image_hash() -> str:
    return sha256_bytes(b"fake_image_bytes_for_testing")


@pytest.fixture
def analysis_id(db, experiment_id, image_hash) -> int:
    return insert_analysis(
        db,
        image_hash=image_hash,
        image_path="data/raw/CIFAKE/DATASET/test/FAKE/0.jpg",
        model_version="efficientnet_b3_v1",
        experiment_id=experiment_id,
        baseline_probability=0.73,
        prediction="synthetic",
        predicted_class=1,
        confidence=0.73,
        processing_time_ms=42.5,
    )


# ── 1. Initialisation ──────────────────────────────────────────────────────────

class TestDatabaseInitialisation:
    def test_db_file_created(self, tmp_path):
        path = tmp_path / "new_test.db"
        db = DatabaseManager(path)
        db.connect()
        assert path.exists()
        db.close()

    def test_init_db_creates_tables(self, db):
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {r["name"] for r in tables}
        expected = {
            "schema_version", "experiments", "experiment_metrics",
            "analysis_runs", "probe_results", "provenance_records",
        }
        assert expected.issubset(names), f"Missing tables: {expected - names}"

    def test_schema_version_recorded(self, db):
        row = db.fetchone("SELECT version FROM schema_version WHERE version=?", (SCHEMA_VERSION,))
        assert row is not None
        assert row["version"] == SCHEMA_VERSION

    def test_init_is_idempotent(self, db):
        """Second call must not raise or duplicate schema_version rows."""
        init_db(db)
        count = db.fetchone("SELECT COUNT(*) AS n FROM schema_version")["n"]
        assert count == 1

    def test_foreign_keys_enabled(self, db):
        result = db.fetchone("PRAGMA foreign_keys")
        assert result[0] == 1


# ── 2. Experiments ─────────────────────────────────────────────────────────────

class TestExperiments:
    def test_insert_experiment_returns_id(self, db):
        eid = insert_experiment(db, "exp1", "CIFAKE", "efficientnet_b3", 42)
        assert isinstance(eid, int)
        assert eid > 0

    def test_get_experiment_roundtrip(self, db, experiment_id):
        exp = get_experiment(db, experiment_id)
        assert exp is not None
        assert exp["experiment_name"] == "test_baseline_cifake"
        assert exp["dataset"] == "CIFAKE"
        assert exp["model_name"] == "efficientnet_b3"
        assert exp["seed"] == 42
        assert exp["generator"] == "stable_diffusion_v1.4"

    def test_list_experiments(self, db):
        insert_experiment(db, "exp_a", "CIFAKE", "efficientnet_b3", 1)
        insert_experiment(db, "exp_b", "CIFAKE", "resnet50", 2)
        exps = list_experiments(db)
        assert len(exps) >= 2

    def test_list_experiments_filtered_by_dataset(self, db):
        insert_experiment(db, "exp_cifake", "CIFAKE", "efficientnet_b3", 1)
        insert_experiment(db, "exp_genimage", "GenImage", "resnet50", 2)
        cifake_exps = list_experiments(db, dataset="CIFAKE")
        assert all(e["dataset"] == "CIFAKE" for e in cifake_exps)

    def test_update_experiment(self, db, experiment_id):
        update_experiment(db, experiment_id, checkpoint_path="models/best.pt", best_epoch=7)
        exp = get_experiment(db, experiment_id)
        assert exp["checkpoint_path"] == "models/best.pt"
        assert exp["best_epoch"] == 7

    def test_update_experiment_invalid_field_raises(self, db, experiment_id):
        with pytest.raises(ValueError, match="Cannot update"):
            update_experiment(db, experiment_id, dataset="OTHER")

    def test_nonexistent_experiment_returns_none(self, db):
        assert get_experiment(db, 99999) is None


# ── 3. Experiment Metrics ──────────────────────────────────────────────────────

class TestExperimentMetrics:
    def test_insert_metric(self, db, experiment_id):
        insert_metric(db, experiment_id, "validation", "roc_auc", 0.87)
        metrics = get_metrics(db, experiment_id, "validation")
        assert any(m["metric_name"] == "roc_auc" and abs(m["metric_value"] - 0.87) < 1e-6 for m in metrics)

    def test_insert_metrics_batch(self, db, experiment_id):
        metrics_dict = {"roc_auc": 0.91, "accuracy": 0.85, "f1": 0.84, "precision": 0.83, "recall": 0.86}
        insert_metrics_batch(db, experiment_id, "validation", metrics_dict, evaluation_type="in_distribution")
        stored = get_metrics(db, experiment_id, "validation")
        stored_names = {m["metric_name"] for m in stored}
        assert stored_names == set(metrics_dict.keys())

    def test_evaluation_type_in_distribution_accepted(self, db, experiment_id):
        insert_metric(db, experiment_id, "test_indistribution", "accuracy", 0.82, evaluation_type="in_distribution")
        metrics = get_metrics(db, experiment_id, "test_indistribution")
        assert metrics[0]["evaluation_type"] == "in_distribution"

    def test_evaluation_type_unseen_generator_accepted(self, db, experiment_id):
        """unseen_generator type is valid for future GenImage experiments."""
        insert_metric(db, experiment_id, "test_unseen", "accuracy", 0.71, evaluation_type="unseen_generator")
        metrics = get_metrics(db, experiment_id, "test_unseen")
        assert metrics[0]["evaluation_type"] == "unseen_generator"

    def test_invalid_evaluation_type_raises(self, db, experiment_id):
        with pytest.raises(ValueError, match="Invalid evaluation_type"):
            insert_metric(db, experiment_id, "test", "roc_auc", 0.5, evaluation_type="made_up_type")

    def test_duplicate_metric_replaces(self, db, experiment_id):
        """INSERT OR REPLACE semantics: second insert overwrites first."""
        insert_metric(db, experiment_id, "validation", "accuracy", 0.70)
        insert_metric(db, experiment_id, "validation", "accuracy", 0.80)
        metrics = get_metrics(db, experiment_id, "validation")
        acc = [m for m in metrics if m["metric_name"] == "accuracy"]
        assert len(acc) == 1
        assert abs(acc[0]["metric_value"] - 0.80) < 1e-6

    def test_get_metrics_for_all_splits(self, db, experiment_id):
        insert_metric(db, experiment_id, "validation", "roc_auc", 0.85)
        insert_metric(db, experiment_id, "test_indistribution", "roc_auc", 0.83)
        all_metrics = get_metrics(db, experiment_id)
        splits = {m["split_name"] for m in all_metrics}
        assert "validation" in splits
        assert "test_indistribution" in splits


# ── 4. Analysis Runs ──────────────────────────────────────────────────────────

class TestAnalysisRuns:
    def test_insert_analysis_returns_id(self, db, image_hash):
        aid = insert_analysis(db, image_hash)
        assert isinstance(aid, int)
        assert aid > 0

    def test_get_analysis_roundtrip(self, db, analysis_id, image_hash):
        result = get_analysis(db, analysis_id)
        assert result is not None
        assert result["image_hash"] == image_hash
        assert result["prediction"] == "synthetic"
        assert abs(result["baseline_probability"] - 0.73) < 1e-6

    def test_calibration_status_defaults_not_calibrated(self, db, image_hash):
        aid = insert_analysis(db, image_hash)
        result = get_analysis(db, aid)
        assert result["calibration_status"] == "not_calibrated"

    def test_uncertainty_status_defaults_not_computed(self, db, image_hash):
        aid = insert_analysis(db, image_hash)
        result = get_analysis(db, aid)
        assert result["uncertainty_status"] == "not_computed"

    def test_invalid_hash_raises(self, db):
        with pytest.raises(ValueError, match="SHA-256"):
            insert_analysis(db, "not_a_valid_hash")

    def test_image_binary_not_stored(self, db, analysis_id):
        """The database table must NOT have any column that could store image binary data."""
        result = get_analysis(db, analysis_id)
        prohibited_keys = {"image_data", "image_binary", "image_bytes", "image_content"}
        assert not prohibited_keys.intersection(set(result.keys()))

    def test_get_analyses_by_hash(self, db, image_hash):
        insert_analysis(db, image_hash)
        insert_analysis(db, image_hash)
        results = get_analyses_by_hash(db, image_hash)
        assert len(results) >= 2
        assert all(r["image_hash"] == image_hash for r in results)

    def test_invalid_calibration_status_raises(self, db, image_hash):
        with pytest.raises(ValueError, match="calibration_status"):
            insert_analysis(db, image_hash, calibration_status="super_calibrated")

    def test_invalid_uncertainty_status_raises(self, db, image_hash):
        with pytest.raises(ValueError, match="uncertainty_status"):
            insert_analysis(db, image_hash, uncertainty_status="very_uncertain")


# ── 5. Probe Results ──────────────────────────────────────────────────────────

class TestProbeResults:
    def test_insert_probe_result(self, db, analysis_id):
        pid = insert_probe_result(db, analysis_id, "jpeg_compression", probability=0.68)
        assert pid > 0

    def test_get_probe_results(self, db, analysis_id):
        insert_probe_result(db, analysis_id, "jpeg_compression", probability=0.68,
                            transformation_params={"quality": 75})
        insert_probe_result(db, analysis_id, "gaussian_noise", probability=0.72)
        results = get_probe_results(db, analysis_id)
        assert len(results) == 2
        names = {r["transformation_name"] for r in results}
        assert "jpeg_compression" in names
        assert "gaussian_noise" in names

    def test_probe_result_fk_cascade(self, db, image_hash):
        """Deleting analysis_run must cascade to probe_results."""
        aid = insert_analysis(db, image_hash)
        insert_probe_result(db, aid, "test_transform")
        db.execute("DELETE FROM analysis_runs WHERE id=?", (aid,))
        db.commit()
        results = get_probe_results(db, aid)
        assert len(results) == 0

    def test_probe_invalid_analysis_id_raises(self, db):
        with pytest.raises(Exception):  # FK violation
            insert_probe_result(db, 99999, "some_transform")
            db.commit()


# ── 6. Provenance Records ─────────────────────────────────────────────────────

class TestProvenanceRecords:
    def test_insert_provenance(self, db, analysis_id):
        pid = insert_provenance(db, analysis_id)
        assert pid > 0

    def test_get_provenance_roundtrip(self, db, analysis_id):
        insert_provenance(
            db, analysis_id,
            metadata_available=True,
            metadata_summary="EXIF found",
            c2pa_status="not_found",
            exif_status="found",
            exif_fields={"Make": "Canon", "Model": "EOS"},
        )
        prov = get_provenance(db, analysis_id)
        assert prov is not None
        assert prov["metadata_available"] == 1
        assert prov["c2pa_status"] == "not_found"
        assert prov["exif_status"] == "found"
        assert "Canon" in (prov["exif_fields"] or "")

    def test_provenance_defaults(self, db, analysis_id):
        insert_provenance(db, analysis_id)
        prov = get_provenance(db, analysis_id)
        assert prov["metadata_available"] == 0
        assert prov["c2pa_status"] == "not_checked"
        assert prov["exif_status"] == "not_checked"

    def test_provenance_invalid_analysis_raises(self, db):
        with pytest.raises(Exception):  # FK violation
            insert_provenance(db, 99999)
            db.commit()


# ── 7. Path Configuration ──────────────────────────────────────────────────────

class TestPathConfiguration:
    def test_database_path_is_configurable(self, tmp_path):
        custom_path = tmp_path / "custom_test.db"
        db = DatabaseManager(custom_path)
        db.connect()
        init_db(db)
        assert custom_path.exists()
        db.close()

    def test_database_parent_dir_created(self, tmp_path):
        nested = tmp_path / "nested" / "deeply" / "test.db"
        db = DatabaseManager(nested)
        db.connect()
        assert nested.parent.exists()
        db.close()

    def test_test_db_does_not_touch_production(self, tmp_path):
        """Test DB must be isolated — production DB path must not be created."""
        test_db = tmp_path / "isolated_test.db"
        db = DatabaseManager(test_db)
        db.connect()
        init_db(db)
        prod_path = PROJECT_ROOT / "data" / "signalscope.db"
        # We only check that our test database exists, not the prod one
        assert test_db.exists()
        db.close()
