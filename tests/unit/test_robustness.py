"""
Tests for src/robustness/probes.py and src/robustness/stress_test.py

Coverage:
- ProbeConfig and ProbeResult dataclasses
- All 5 built-in probes execute correctly on a synthetic image
- ProbeRunner.run() returns one result per probe
- Failed probes are reported with success=False and no fabricated probability
- StressTestEngine returns correct aggregate statistics
- Stability is None when < 2 successful probes
- Stop reason is threshold_not_configured when thresholds are null
- No probe mutates the original image
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.robustness.probes import (
    BUILTIN_PROBES,
    ProbeConfig,
    ProbeResult,
    ProbeRunner,
    _jpeg_encode_decode,
    _resize_and_restore,
)
from src.robustness.stress_test import (
    StressTestConfig,
    StressTestEngine,
    StressTestResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_image_32():
    """A simple 32×32 RGB image for testing."""
    img = Image.new("RGB", (32, 32), color=(120, 80, 40))
    return img


@pytest.fixture
def rgb_image_224():
    """A 224×224 RGB image."""
    img = Image.new("RGB", (224, 224), color=(60, 120, 180))
    return img


def _make_fake_runner(return_prob: float = 0.7, calibration_temperature: float = None):
    """Create a ProbeRunner with a mocked model that always returns return_prob."""
    model = MagicMock()
    import torch

    # Create a fake transform that returns a real tensor
    def fake_transform(img):
        return torch.zeros(3, 224, 224)

    runner = ProbeRunner(
        model=model,
        val_transform=fake_transform,
        device=torch.device("cpu"),
        calibration_temperature=calibration_temperature,
        calibration_status="calibrated" if calibration_temperature else "not_calibrated",
    )

    # Override _infer to return a fixed probability
    runner._infer = MagicMock(return_value=return_prob)
    return runner


# ---------------------------------------------------------------------------
# Probe transformation tests
# ---------------------------------------------------------------------------

class TestProbeTransformations:

    def test_jpeg_q90_preserves_size(self, rgb_image_32):
        result = _jpeg_encode_decode(rgb_image_32, quality=90)
        assert result.size == rgb_image_32.size

    def test_jpeg_q70_preserves_size(self, rgb_image_32):
        result = _jpeg_encode_decode(rgb_image_32, quality=70)
        assert result.size == rgb_image_32.size

    def test_jpeg_q50_preserves_size(self, rgb_image_32):
        result = _jpeg_encode_decode(rgb_image_32, quality=50)
        assert result.size == rgb_image_32.size

    def test_jpeg_returns_rgb(self, rgb_image_32):
        result = _jpeg_encode_decode(rgb_image_32, quality=70)
        assert result.mode == "RGB"

    def test_resize_75pct_preserves_original_size(self, rgb_image_224):
        result = _resize_and_restore(rgb_image_224, scale_pct=75)
        assert result.size == rgb_image_224.size

    def test_resize_50pct_preserves_original_size(self, rgb_image_224):
        result = _resize_and_restore(rgb_image_224, scale_pct=50)
        assert result.size == rgb_image_224.size

    def test_probe_does_not_mutate_original(self, rgb_image_32):
        original_data = list(rgb_image_32.getdata())
        _jpeg_encode_decode(rgb_image_32, quality=50)
        assert list(rgb_image_32.getdata()) == original_data

    def test_resize_does_not_mutate_original(self, rgb_image_224):
        original_size = rgb_image_224.size
        _resize_and_restore(rgb_image_224, scale_pct=50)
        assert rgb_image_224.size == original_size

    def test_jpeg_changes_pixels_at_low_quality(self, rgb_image_32):
        """Lossy JPEG should alter pixel values at Q50."""
        original = np.array(rgb_image_32)
        transformed = np.array(_jpeg_encode_decode(rgb_image_32, quality=50))
        # Not all pixels need to change but some should for a non-uniform image
        # Use a gradient image to guarantee change
        grad_img = Image.fromarray(
            np.arange(0, 32*32, dtype=np.uint8).reshape(32, 32, 1).repeat(3, axis=2)
        )
        orig_arr = np.array(grad_img)
        trans_arr = np.array(_jpeg_encode_decode(grad_img, quality=50))
        # JPEG at Q50 on a gradient will modify at least some pixels
        assert not np.array_equal(orig_arr, trans_arr)


class TestBuiltinProbes:

    def test_builtin_probes_count(self):
        assert len(BUILTIN_PROBES) == 5

    def test_builtin_probe_names_unique(self):
        names = [p.name for p in BUILTIN_PROBES]
        assert len(names) == len(set(names))

    def test_builtin_probes_have_parameters(self):
        for probe in BUILTIN_PROBES:
            assert isinstance(probe.parameters, dict)
            assert len(probe.parameters) > 0

    def test_builtin_probes_have_descriptions(self):
        for probe in BUILTIN_PROBES:
            assert isinstance(probe.description, str)
            assert len(probe.description) > 0

    def test_builtin_probe_order_is_deterministic(self):
        """Order must be stable across multiple accesses."""
        names_first  = [p.name for p in BUILTIN_PROBES]
        names_second = [p.name for p in BUILTIN_PROBES]
        assert names_first == names_second

    def test_jpeg_probes_have_quality_param(self):
        jpeg_probes = [p for p in BUILTIN_PROBES if p.name.startswith("jpeg_")]
        for p in jpeg_probes:
            assert "quality" in p.parameters

    def test_resize_probes_have_scale_param(self):
        resize_probes = [p for p in BUILTIN_PROBES if p.name.startswith("resize_")]
        for p in resize_probes:
            assert "scale_pct" in p.parameters


# ---------------------------------------------------------------------------
# ProbeRunner tests
# ---------------------------------------------------------------------------

class TestProbeRunner:

    def test_run_returns_one_result_per_probe(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.6)
        results = runner.run(rgb_image_224, probes=BUILTIN_PROBES)
        assert len(results) == len(BUILTIN_PROBES)

    def test_run_all_succeed_with_mocked_model(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.75)
        results = runner.run(rgb_image_224)
        for r in results:
            assert r.success is True
            assert r.error is None
            assert r.raw_probability is not None

    def test_probe_result_names_match_probe_configs(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.5)
        results = runner.run(rgb_image_224)
        for result, probe in zip(results, BUILTIN_PROBES):
            assert result.probe_name == probe.name

    def test_failed_probe_has_no_probability(self):
        """A probe whose transformation crashes must NOT fabricate a probability."""
        runner = _make_fake_runner(return_prob=0.5)

        def crashing_transform(img):
            raise RuntimeError("Simulated transformation failure")

        bad_probe = ProbeConfig(
            name="failing_probe",
            transform_fn=crashing_transform,
            description="Always fails in transform",
            parameters={"test": True},
        )
        result = runner.run_probe(Image.new("RGB", (32, 32)), bad_probe)
        assert result.success is False
        assert result.raw_probability is None
        assert result.error is not None
        assert "Simulated" in result.error

    def test_calibration_applied_when_temperature_set(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.7, calibration_temperature=2.0)
        results = runner.run(rgb_image_224, probes=[BUILTIN_PROBES[0]])
        r = results[0]
        assert r.success is True
        assert r.calibrated_probability is not None
        assert r.calibration_status == "calibrated"
        # Calibrated probability must be lower than raw due to T > 1
        assert r.calibrated_probability < r.raw_probability

    def test_no_calibration_when_temperature_not_set(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.7, calibration_temperature=None)
        results = runner.run(rgb_image_224, probes=[BUILTIN_PROBES[0]])
        r = results[0]
        assert r.calibrated_probability is None

    def test_processing_time_is_recorded(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.5)
        results = runner.run(rgb_image_224, probes=[BUILTIN_PROBES[0]])
        assert results[0].processing_time_s is not None
        assert results[0].processing_time_s >= 0

    def test_probe_result_to_dict(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.6)
        results = runner.run(rgb_image_224, probes=[BUILTIN_PROBES[0]])
        d = results[0].to_dict()
        assert "probe_name" in d
        assert "success" in d
        assert "raw_probability" in d
        assert "calibrated_probability" in d
        assert "calibration_status" in d
        assert "processing_time_s" in d


# ---------------------------------------------------------------------------
# StressTestEngine tests
# ---------------------------------------------------------------------------

class TestStressTestEngine:

    def _make_engine(self, return_prob=0.7, temperature=None, config=None):
        runner = _make_fake_runner(return_prob=return_prob, calibration_temperature=temperature)
        return StressTestEngine(probe_runner=runner, config=config)

    def test_returns_stress_test_result(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert isinstance(result, StressTestResult)

    def test_baseline_probability_matches_mock(self, rgb_image_224):
        engine = self._make_engine(return_prob=0.8)
        result = engine.run(rgb_image_224)
        assert abs(result.baseline_raw_probability - 0.8) < 1e-5

    def test_probes_configured_matches_builtin_count(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert result.probes_configured == len(BUILTIN_PROBES)

    def test_all_probes_run_when_no_threshold(self, rgb_image_224):
        engine = self._make_engine(return_prob=0.6)
        result = engine.run(rgb_image_224)
        assert result.probes_run == len(BUILTIN_PROBES)

    def test_stop_reason_threshold_not_configured_by_default(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert result.stop_reason == "threshold_not_configured"
        assert result.threshold_configured is False

    def test_stop_reason_all_complete_when_threshold_set(self, rgb_image_224):
        config = StressTestConfig(stability_threshold=0.9)
        engine = self._make_engine(config=config)
        result = engine.run(rgb_image_224)
        assert result.stop_reason == "all_probes_complete"
        assert result.threshold_configured is True

    def test_mean_probability_computed(self, rgb_image_224):
        engine = self._make_engine(return_prob=0.7)
        result = engine.run(rgb_image_224)
        assert result.mean_probability is not None
        # All probes return same prob so mean == prob
        assert abs(result.mean_probability - 0.7) < 1e-3

    def test_std_probability_none_with_single_probe(self, rgb_image_224):
        config = StressTestConfig(probes=[BUILTIN_PROBES[0]])
        engine = self._make_engine(config=config)
        result = engine.run(rgb_image_224)
        # Only 1 successful probe → std requires ddof=1 → None
        assert result.std_probability is None
        assert result.stability is None

    def test_stability_computed_with_two_or_more_probes(self, rgb_image_224):
        config = StressTestConfig(probes=BUILTIN_PROBES[:2])
        engine = self._make_engine(return_prob=0.7, config=config)
        result = engine.run(rgb_image_224)
        # Both return same prob → std=0, stability=1.0
        assert result.std_probability is not None
        assert result.stability is not None
        assert result.stability >= 0.0
        assert result.stability <= 1.0

    def test_evidence_sufficient_requires_two_successes(self, rgb_image_224):
        # Only 1 probe configured → evidence_sufficient=False
        config = StressTestConfig(probes=[BUILTIN_PROBES[0]])
        engine = self._make_engine(config=config)
        result = engine.run(rgb_image_224)
        assert result.evidence_sufficient is False

    def test_evidence_sufficient_true_with_five_probes(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert result.evidence_sufficient is True

    def test_failed_probes_excluded_from_statistics(self, rgb_image_224):
        runner = _make_fake_runner(return_prob=0.6)
        # Make all probe inferences fail
        runner._infer = MagicMock(side_effect=[0.6] + [RuntimeError("boom")] * 5)
        engine = StressTestEngine(probe_runner=runner)
        result = engine.run(rgb_image_224)
        assert result.probes_failed > 0
        # Mean/std only from successful probes
        assert result.probes_successful == 0
        assert result.mean_probability is None

    def test_final_prediction_real_below_threshold(self, rgb_image_224):
        engine = self._make_engine(return_prob=0.3)
        result = engine.run(rgb_image_224)
        assert result.final_prediction == "real"

    def test_final_prediction_synthetic_above_threshold(self, rgb_image_224):
        engine = self._make_engine(return_prob=0.8)
        result = engine.run(rgb_image_224)
        assert result.final_prediction == "synthetic"

    def test_to_dict_has_required_keys(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        d = result.to_dict()
        required = [
            "baseline_raw_probability", "probes_run", "probes_successful",
            "probes_failed", "mean_probability", "std_probability",
            "stability", "final_prediction", "final_probability",
            "stop_reason", "threshold_configured", "evidence_sufficient",
            "probe_results", "total_time_s",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_probe_results_list_matches_probes_run(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert len(result.probe_results) == result.probes_run

    def test_total_time_recorded(self, rgb_image_224):
        engine = self._make_engine()
        result = engine.run(rgb_image_224)
        assert result.total_time_s >= 0
