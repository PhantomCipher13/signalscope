"""
tests/unit/test_calibration.py
================================
Unit tests for Phase 3 calibration module.

Tests use deterministic controlled fixtures — NOT real model predictions.
These tests verify SOFTWARE CORRECTNESS only.
Do not report fixture ECE/Brier as model performance.

Fixtures:
  - perfect_probs:   probs == labels (ideal, ECE≈0, Brier≈0)
  - uniform_probs:   all probs = 0.5 (maximally uncertain)
  - stable_probs:    [0.90, 0.91, 0.89, 0.90, 0.91] (low std)
  - unstable_probs:  [0.90, 0.65, 0.82, 0.57, 0.76] (high std)
  - single_prob:     [0.90] (insufficient evidence)
  - overconfident:   probs all 0.99 regardless of label
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.status import CalibrationStatus, CalibrationResult
from src.calibration.metrics import compute_ece, compute_brier, reliability_diagram_data
from src.calibration.temperature_scaler import TemperatureScaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_labels():
    """Balanced binary labels for 100 samples."""
    return np.array([0] * 50 + [1] * 50, dtype=np.int32)

@pytest.fixture
def perfect_probs():
    """Probabilities that exactly match labels (ECE ≈ 0, Brier ≈ 0)."""
    return np.array([0.01] * 50 + [0.99] * 50, dtype=np.float32)

@pytest.fixture
def uniform_probs():
    """All probs = 0.5 — maximally uncertain."""
    return np.full(100, 0.5, dtype=np.float32)

@pytest.fixture
def overconfident_probs():
    """All probs = 0.95 regardless of label — high ECE."""
    return np.full(100, 0.95, dtype=np.float32)

@pytest.fixture
def stable_probs_list():
    return [0.90, 0.91, 0.89, 0.90, 0.91]

@pytest.fixture
def unstable_probs_list():
    return [0.90, 0.65, 0.82, 0.57, 0.76]

@pytest.fixture
def single_prob_list():
    return [0.90]


# ── 1. CalibrationStatus ──────────────────────────────────────────────────────

class TestCalibrationStatus:
    def test_all_states_defined(self):
        assert CalibrationStatus.NOT_CALIBRATED.value == "not_calibrated"
        assert CalibrationStatus.CALIBRATED.value == "calibrated"
        assert CalibrationStatus.CALIBRATION_UNAVAILABLE.value == "calibration_unavailable"
        assert CalibrationStatus.CALIBRATION_FALLBACK.value == "calibration_fallback"

    def test_status_is_string_enum(self):
        assert isinstance(CalibrationStatus.NOT_CALIBRATED, str)

    def test_calibration_result_to_dict(self):
        result = CalibrationResult(
            temperature=1.2,
            status=CalibrationStatus.CALIBRATED,
            fitted_on="validation",
            n_samples_fitted=1000,
            ece_before=0.08,
            ece_after=0.04,
            brier_before=0.12,
            brier_after=0.09,
        )
        d = result.to_dict()
        assert d["status"] == "calibrated"
        assert d["temperature"] == 1.2
        assert d["fitted_on"] == "validation"
        assert d["n_samples_fitted"] == 1000

    def test_calibration_result_none_fields_allowed(self):
        """All metric fields can be None when calibration hasn't run."""
        result = CalibrationResult(
            temperature=None,
            status=CalibrationStatus.CALIBRATION_UNAVAILABLE,
            fitted_on=None,
            n_samples_fitted=None,
            ece_before=None,
            ece_after=None,
            brier_before=None,
            brier_after=None,
        )
        d = result.to_dict()
        assert d["temperature"] is None
        assert d["ece_before"] is None


# ── 2. ECE ────────────────────────────────────────────────────────────────────

class TestECE:
    def test_perfect_calibration_low_ece(self, binary_labels, perfect_probs):
        ece = compute_ece(binary_labels, perfect_probs)
        assert 0.0 <= ece <= 0.05, f"Perfect calibration ECE should be near 0, got {ece}"

    def test_overconfident_high_ece(self, binary_labels, overconfident_probs):
        ece = compute_ece(binary_labels, overconfident_probs)
        assert ece > 0.3, f"Overconfident predictor should have high ECE, got {ece}"

    def test_ece_is_bounded(self, binary_labels, uniform_probs):
        ece = compute_ece(binary_labels, uniform_probs)
        assert 0.0 <= ece <= 1.0

    def test_ece_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_ece(np.array([]), np.array([]))

    def test_ece_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            compute_ece(np.array([0, 1, 0]), np.array([0.5, 0.5]))

    def test_ece_returns_float(self, binary_labels, uniform_probs):
        ece = compute_ece(binary_labels, uniform_probs)
        assert isinstance(ece, float)

    def test_ece_configurable_bins(self, binary_labels, perfect_probs):
        ece_10 = compute_ece(binary_labels, perfect_probs, n_bins=10)
        ece_20 = compute_ece(binary_labels, perfect_probs, n_bins=20)
        # Both should be valid floats regardless of bin count
        assert isinstance(ece_10, float)
        assert isinstance(ece_20, float)

    def test_ece_single_class_does_not_crash(self):
        """All-same labels should not crash ECE computation."""
        labels = np.zeros(20, dtype=np.int32)
        probs = np.full(20, 0.2, dtype=np.float32)
        ece = compute_ece(labels, probs)
        assert isinstance(ece, float)


# ── 3. Brier Score ────────────────────────────────────────────────────────────

class TestBrierScore:
    def test_perfect_brier_near_zero(self, binary_labels, perfect_probs):
        brier = compute_brier(binary_labels, perfect_probs)
        assert brier < 0.01, f"Perfect predictor Brier should be near 0, got {brier}"

    def test_worst_brier_near_one(self, binary_labels):
        worst_probs = np.where(binary_labels == 0, 1.0, 0.0).astype(np.float32)
        brier = compute_brier(binary_labels, worst_probs)
        assert brier > 0.9, f"Inverted predictor Brier should be near 1, got {brier}"

    def test_uniform_brier(self, binary_labels, uniform_probs):
        brier = compute_brier(binary_labels, uniform_probs)
        # uniform = 0.5 for balanced labels → Brier = 0.25
        assert abs(brier - 0.25) < 0.01, f"Uniform predictor Brier should be ~0.25, got {brier}"

    def test_brier_empty_raises(self):
        with pytest.raises(ValueError):
            compute_brier(np.array([]), np.array([]))

    def test_brier_returns_float(self, binary_labels, uniform_probs):
        brier = compute_brier(binary_labels, uniform_probs)
        assert isinstance(brier, float)

    def test_brier_is_bounded(self, binary_labels, uniform_probs):
        brier = compute_brier(binary_labels, uniform_probs)
        assert 0.0 <= brier <= 1.0


# ── 4. Reliability Diagram Data ───────────────────────────────────────────────

class TestReliabilityDiagram:
    def test_returns_required_keys(self, binary_labels, uniform_probs):
        result = reliability_diagram_data(binary_labels, uniform_probs)
        assert "bin_centers" in result
        assert "mean_confidence" in result
        assert "fraction_positive" in result
        assert "bin_counts" in result

    def test_bin_centers_length(self, binary_labels, uniform_probs):
        result = reliability_diagram_data(binary_labels, uniform_probs, n_bins=10)
        assert len(result["bin_centers"]) == 10

    def test_bin_counts_sum_to_n(self, binary_labels, uniform_probs):
        result = reliability_diagram_data(binary_labels, uniform_probs)
        assert result["bin_counts"].sum() == len(binary_labels)


# ── 5. TemperatureScaler ─────────────────────────────────────────────────────

class TestTemperatureScaler:
    def test_initial_state_not_fitted(self):
        scaler = TemperatureScaler()
        assert not scaler.is_fitted
        assert scaler.calibration_status == CalibrationStatus.NOT_CALIBRATED
        assert scaler.temperature == 1.0

    def test_fit_on_validation_sets_fitted(self, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        # Either calibrated or fallback — not unavailable for sufficient data
        assert result.status in (
            CalibrationStatus.CALIBRATED,
            CalibrationStatus.CALIBRATION_FALLBACK,
        )
        assert scaler.is_fitted

    def test_fit_returns_calibration_result(self, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        assert isinstance(result, CalibrationResult)
        assert result.fitted_on == "validation"
        assert result.n_samples_fitted == len(binary_labels)

    def test_fit_computes_ece_before_and_after(self, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        # ECE before must be non-None for sufficient data
        assert result.ece_before is not None
        # ECE after computed when fitting succeeds
        if result.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.CALIBRATION_FALLBACK):
            assert result.ece_after is not None

    def test_fit_single_sample_unavailable(self):
        """One sample cannot support calibration — must return UNAVAILABLE."""
        scaler = TemperatureScaler()
        result = scaler.fit(np.array([0.9]), np.array([1]), split="validation")
        assert result.status == CalibrationStatus.CALIBRATION_UNAVAILABLE
        assert result.temperature is None
        assert not scaler.is_fitted

    def test_fit_empty_unavailable(self):
        scaler = TemperatureScaler()
        result = scaler.fit(np.array([]), np.array([]), split="validation")
        assert result.status == CalibrationStatus.CALIBRATION_UNAVAILABLE

    def test_transform_unfitted_returns_input_unchanged(self):
        scaler = TemperatureScaler()
        probs = np.array([0.3, 0.7, 0.5])
        out = scaler.transform(probs)
        np.testing.assert_array_almost_equal(out, probs)

    def test_transform_fitted_outputs_in_range(self, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        if result.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.CALIBRATION_FALLBACK):
            out = scaler.transform(overconfident_probs)
            assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_transform_with_status_returns_tuple(self, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        scaler.fit(overconfident_probs, binary_labels, split="validation")
        probs_out, status = scaler.transform_with_status(overconfident_probs)
        assert isinstance(probs_out, np.ndarray)
        assert isinstance(status, CalibrationStatus)

    def test_unfitted_transform_with_status_returns_not_calibrated(self):
        scaler = TemperatureScaler()
        _, status = scaler.transform_with_status(np.array([0.7, 0.8]))
        assert status == CalibrationStatus.NOT_CALIBRATED

    def test_calibration_never_silently_called_calibrated(self):
        """A newly created scaler must never report CALIBRATED."""
        scaler = TemperatureScaler()
        assert scaler.calibration_status != CalibrationStatus.CALIBRATED

    def test_save_load_roundtrip(self, tmp_path, binary_labels, overconfident_probs):
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        save_path = tmp_path / "calibration.json"
        scaler.save(str(save_path))
        assert save_path.exists()

        loaded = TemperatureScaler.load(str(save_path))
        assert isinstance(loaded, TemperatureScaler)
        if result.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.CALIBRATION_FALLBACK):
            assert loaded.is_fitted
            assert abs(loaded.temperature - scaler.temperature) < 1e-6

    def test_load_validates_temperature_positive(self, tmp_path):
        bad_data = {"temperature": -0.5, "is_fitted": True, "_fitted_on_split": "validation"}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError, match="Temperature"):
            TemperatureScaler.load(str(path))

    def test_temperature_softens_overconfident_predictions(self, binary_labels, overconfident_probs):
        """Temperature scaling should push overconfident probs toward 0.5."""
        scaler = TemperatureScaler()
        result = scaler.fit(overconfident_probs, binary_labels, split="validation")
        if result.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.CALIBRATION_FALLBACK):
            out = scaler.transform(overconfident_probs)
            # Calibrated probs should be less extreme than input probs
            assert out.max() < overconfident_probs.max() + 0.001

    def test_fit_accepts_logits(self, binary_labels):
        """TemperatureScaler should accept raw logits (outside [0,1])."""
        logits = np.array([3.5] * 50 + [-3.5] * 50, dtype=np.float32)
        # Swap: high logits = synthetic (1), low logits = real (0)
        scaler = TemperatureScaler()
        result = scaler.fit(logits, binary_labels, split="validation")
        assert result.n_samples_fitted == 100
