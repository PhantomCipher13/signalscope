"""
tests/unit/test_reliability.py
================================
Unit tests for Phase 3 reliability engine.

Tests use deterministic controlled fixtures — NOT real model predictions.
These are SOFTWARE CORRECTNESS tests only.

Key invariants tested:
  1. Single observation → stability=None, status=INSUFFICIENT_EVIDENCE
     (Do NOT return stability=1.0 for std([x])=0)
  2. Stable sequence → lower std, status=STABLE (when threshold configured)
  3. Unstable sequence → higher std, status=UNSTABLE (when threshold configured)
  4. Unconfigured thresholds → THRESHOLD_NOT_CONFIGURED
  5. calibration_status passes through correctly
  6. Aggregator correctly handles edge cases

Controlled fixtures (software tests, NOT model evaluations):
  stable_probs:   [0.90, 0.91, 0.89, 0.90, 0.91]  std ≈ 0.0075
  unstable_probs: [0.90, 0.65, 0.82, 0.57, 0.76]  std ≈ 0.135
  single_prob:    [0.90]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability.status import ReliabilityStatus, StabilityStatus, ReliabilityResult
from src.reliability.aggregator import ProbabilityAggregator
from src.reliability.engine import ReliabilityEngine, format_result


# ── Fixtures ──────────────────────────────────────────────────────────────────

STABLE_PROBS   = [0.90, 0.91, 0.89, 0.90, 0.91]   # std ≈ 0.0075
UNSTABLE_PROBS = [0.90, 0.65, 0.82, 0.57, 0.76]   # std ≈ 0.135
SINGLE_PROB    = [0.90]

@pytest.fixture
def engine_no_config():
    """Engine with no thresholds configured — simulates pre-validation state."""
    return ReliabilityEngine(config={})

@pytest.fixture
def engine_configured():
    """Engine with explicit thresholds set — simulates post-validation state."""
    return ReliabilityEngine(config={
        "minimum_observations": 3,
        "stability_threshold": 0.05,  # std < 0.05 → stable
    })


# ── 1. ReliabilityStatus Enum ──────────────────────────────────────────────────

class TestReliabilityStatus:
    def test_all_states_defined(self):
        assert ReliabilityStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert ReliabilityStatus.STABLE.value == "stable"
        assert ReliabilityStatus.UNSTABLE.value == "unstable"
        assert ReliabilityStatus.UNAVAILABLE.value == "unavailable"
        assert ReliabilityStatus.THRESHOLD_NOT_CONFIGURED.value == "threshold_not_configured"

    def test_stability_status_defined(self):
        assert StabilityStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert StabilityStatus.COMPUTED.value == "computed"
        assert StabilityStatus.UNAVAILABLE.value == "unavailable"

    def test_status_is_string_enum(self):
        assert isinstance(ReliabilityStatus.STABLE, str)

    def test_result_to_dict_contains_all_fields(self):
        result = ReliabilityResult(
            baseline_probability=0.82,
            calibrated_probability=None,
            calibration_status="not_calibrated",
            observation_count=1,
            probabilities=[0.82],
            mean_probability=0.82,
            standard_deviation=None,
            stability=None,
            stability_status=StabilityStatus.INSUFFICIENT_EVIDENCE,
            reliability_status=ReliabilityStatus.INSUFFICIENT_EVIDENCE,
            reliability_note="Only one observation.",
            threshold_configured=False,
        )
        d = result.to_dict()
        assert "baseline_probability" in d
        assert "stability" in d
        assert d["stability"] is None
        assert d["observation_count"] == 1


# ── 2. ProbabilityAggregator ──────────────────────────────────────────────────

class TestProbabilityAggregator:
    def setup_method(self):
        self.agg = ProbabilityAggregator()

    def test_single_observation_no_std(self):
        result = self.agg.aggregate(SINGLE_PROB, minimum_observations=3)
        assert result["observation_count"] == 1
        assert result["standard_deviation"] is None  # n<2, no std
        assert result["stability"] is None           # n < minimum
        assert result["stability_status"] == StabilityStatus.INSUFFICIENT_EVIDENCE

    def test_single_observation_stability_is_not_1(self):
        """
        CRITICAL: std([x])=0 is mathematically true but does NOT mean stable.
        Stability must be None for a single observation.
        """
        result = self.agg.aggregate([0.90], minimum_observations=2)
        assert result["stability"] is None, (
            "stability must be None for a single observation — "
            "std=0 does not prove stability, only that there is one data point."
        )

    def test_stable_sequence_low_std(self):
        result = self.agg.aggregate(STABLE_PROBS, minimum_observations=3)
        assert result["standard_deviation"] is not None
        assert result["standard_deviation"] < 0.02  # should be ~0.0075

    def test_unstable_sequence_high_std(self):
        result = self.agg.aggregate(UNSTABLE_PROBS, minimum_observations=3)
        assert result["standard_deviation"] is not None
        assert result["standard_deviation"] > 0.1  # should be ~0.135

    def test_stable_vs_unstable_std_ordering(self):
        stable_result   = self.agg.aggregate(STABLE_PROBS,   minimum_observations=3)
        unstable_result = self.agg.aggregate(UNSTABLE_PROBS, minimum_observations=3)
        assert stable_result["standard_deviation"] < unstable_result["standard_deviation"]

    def test_stability_clamped_to_unit_interval(self):
        # Even if std > 1 (impossible for probs in [0,1]), stability must be clamped
        result = self.agg.aggregate(STABLE_PROBS, minimum_observations=3)
        if result["stability"] is not None:
            assert 0.0 <= result["stability"] <= 1.0

    def test_no_minimum_observations_returns_note(self):
        result = self.agg.aggregate(STABLE_PROBS, minimum_observations=None)
        assert "not configured" in result["note"].lower() or result["note"] != ""

    def test_insufficient_observations_reports_status(self):
        result = self.agg.aggregate([0.90, 0.91], minimum_observations=5)
        assert result["stability_status"] == StabilityStatus.INSUFFICIENT_EVIDENCE

    def test_mean_always_computed(self):
        result = self.agg.aggregate([0.70, 0.80], minimum_observations=5)
        assert result["mean_probability"] is not None
        assert abs(result["mean_probability"] - 0.75) < 1e-6

    def test_std_uses_ddof_1(self):
        """Verify std is sample std (ddof=1), not population std."""
        import statistics
        result = self.agg.aggregate([0.80, 0.90], minimum_observations=2)
        expected = statistics.stdev([0.80, 0.90])
        assert abs(result["standard_deviation"] - expected) < 1e-9


# ── 3. ReliabilityEngine — single observation ─────────────────────────────────

class TestReliabilityEngineSingle:
    def test_single_returns_insufficient_evidence(self, engine_no_config):
        result = engine_no_config.analyze_single(0.85)
        assert result.reliability_status == ReliabilityStatus.INSUFFICIENT_EVIDENCE

    def test_single_stability_is_none(self, engine_no_config):
        result = engine_no_config.analyze_single(0.85)
        assert result.stability is None

    def test_single_stability_status_is_insufficient(self, engine_no_config):
        result = engine_no_config.analyze_single(0.85)
        assert result.stability_status == StabilityStatus.INSUFFICIENT_EVIDENCE

    def test_single_observation_count_is_1(self, engine_no_config):
        result = engine_no_config.analyze_single(0.85)
        assert result.observation_count == 1

    def test_single_std_is_none(self, engine_no_config):
        result = engine_no_config.analyze_single(0.85)
        assert result.standard_deviation is None

    def test_single_preserves_baseline_probability(self, engine_no_config):
        result = engine_no_config.analyze_single(0.73)
        assert result.baseline_probability == 0.73

    def test_single_passes_calibration_status(self, engine_no_config):
        result = engine_no_config.analyze_single(0.73, calibration_status="calibrated", calibrated_probability=0.68)
        assert result.calibration_status == "calibrated"
        assert result.calibrated_probability == 0.68

    def test_single_not_calibrated_leaves_calibrated_prob_none(self, engine_no_config):
        result = engine_no_config.analyze_single(0.73)
        assert result.calibrated_probability is None

    def test_configured_engine_single_still_insufficient(self, engine_configured):
        """Even with configured thresholds, one observation = INSUFFICIENT_EVIDENCE."""
        result = engine_configured.analyze_single(0.85)
        assert result.reliability_status == ReliabilityStatus.INSUFFICIENT_EVIDENCE
        assert result.stability is None


# ── 4. ReliabilityEngine — unconfigured thresholds ───────────────────────────

class TestReliabilityEngineUnconfigured:
    def test_no_config_probes_returns_threshold_not_configured(self, engine_no_config):
        result = engine_no_config.analyze_probes(0.90, [0.91, 0.89, 0.90])
        assert result.reliability_status == ReliabilityStatus.THRESHOLD_NOT_CONFIGURED

    def test_no_config_threshold_configured_false(self, engine_no_config):
        result = engine_no_config.analyze_probes(0.90, [0.91, 0.89, 0.90])
        assert result.threshold_configured is False

    def test_no_config_still_computes_std(self, engine_no_config):
        """Even without thresholds, std should still be computed when n>=2."""
        result = engine_no_config.analyze_probes(0.90, [0.91, 0.89])
        assert result.standard_deviation is not None

    def test_no_config_still_reports_observation_count(self, engine_no_config):
        result = engine_no_config.analyze_probes(0.90, [0.91, 0.89, 0.88])
        assert result.observation_count == 4  # baseline + 3 probes


# ── 5. ReliabilityEngine — configured, stable sequence ───────────────────────

class TestReliabilityEngineStable:
    def test_stable_sequence_returns_stable(self, engine_configured):
        # STABLE_PROBS std ≈ 0.0075, threshold = 0.05
        baseline = STABLE_PROBS[0]
        probes   = STABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        assert result.reliability_status == ReliabilityStatus.STABLE

    def test_stable_observation_count_correct(self, engine_configured):
        baseline = STABLE_PROBS[0]
        probes   = STABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        assert result.observation_count == len(STABLE_PROBS)

    def test_stable_std_low(self, engine_configured):
        baseline = STABLE_PROBS[0]
        probes   = STABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        assert result.standard_deviation is not None
        assert result.standard_deviation < 0.02

    def test_stable_stability_in_unit_interval(self, engine_configured):
        baseline = STABLE_PROBS[0]
        probes   = STABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        if result.stability is not None:
            assert 0.0 <= result.stability <= 1.0


# ── 6. ReliabilityEngine — configured, unstable sequence ─────────────────────

class TestReliabilityEngineUnstable:
    def test_unstable_sequence_returns_unstable(self, engine_configured):
        # UNSTABLE_PROBS std ≈ 0.135, threshold = 0.05
        baseline = UNSTABLE_PROBS[0]
        probes   = UNSTABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        assert result.reliability_status == ReliabilityStatus.UNSTABLE

    def test_unstable_std_high(self, engine_configured):
        baseline = UNSTABLE_PROBS[0]
        probes   = UNSTABLE_PROBS[1:]
        result = engine_configured.analyze_probes(baseline, probes)
        assert result.standard_deviation > 0.1

    def test_unstable_vs_stable_status_differ(self, engine_configured):
        stable_result = engine_configured.analyze_probes(
            STABLE_PROBS[0], STABLE_PROBS[1:]
        )
        unstable_result = engine_configured.analyze_probes(
            UNSTABLE_PROBS[0], UNSTABLE_PROBS[1:]
        )
        assert stable_result.reliability_status != unstable_result.reliability_status


# ── 7. ReliabilityEngine — calibration flow ──────────────────────────────────

class TestReliabilityEngineCalibration:
    def test_calibration_status_propagates(self, engine_configured):
        result = engine_configured.analyze_probes(
            0.90, [0.91, 0.89, 0.88],
            calibration_status="calibrated",
            calibrated_probability=0.85,
        )
        assert result.calibration_status == "calibrated"
        assert result.calibrated_probability == 0.85

    def test_missing_calibration_leaves_calibrated_prob_none(self, engine_configured):
        result = engine_configured.analyze_probes(0.90, [0.91, 0.89])
        assert result.calibrated_probability is None

    def test_uncalibrated_status_does_not_claim_calibrated(self, engine_configured):
        result = engine_configured.analyze_probes(0.90, [0.91, 0.89])
        assert result.calibration_status == "not_calibrated"


# ── 8. format_result ──────────────────────────────────────────────────────────

class TestFormatResult:
    def test_format_result_returns_string(self, engine_no_config):
        result = engine_no_config.analyze_single(0.82)
        formatted = format_result(result)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_result_contains_status(self, engine_no_config):
        result = engine_no_config.analyze_single(0.82)
        formatted = format_result(result)
        assert "insufficient_evidence" in formatted

    def test_format_result_contains_observation_count(self, engine_no_config):
        result = engine_no_config.analyze_single(0.82)
        formatted = format_result(result)
        assert "1" in formatted  # observation count


# ── 9. Invalid inputs ─────────────────────────────────────────────────────────

class TestInvalidInputs:
    def test_aggregator_empty_list_handled(self):
        agg = ProbabilityAggregator()
        # Empty list should not crash, mean_probability should be None or 0
        try:
            result = agg.aggregate([], minimum_observations=3)
            assert result["observation_count"] == 0
        except (ZeroDivisionError, ValueError):
            pass  # raising is also acceptable

    def test_engine_baseline_prob_stored(self, engine_no_config):
        result = engine_no_config.analyze_single(0.123)
        assert abs(result.baseline_probability - 0.123) < 1e-6

    def test_engine_probes_include_baseline_in_count(self, engine_configured):
        result = engine_configured.analyze_probes(0.90, [0.91, 0.89])
        # baseline + 2 probes = 3 observations
        assert result.observation_count == 3
        assert len(result.probabilities) == 3
