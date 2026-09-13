"""
SignalScope — src/robustness/stress_test.py
=============================================
Adaptive stress-testing engine.

Design:
  1. Run baseline inference on the original image.
  2. Run probes in deterministic order.
  3. Aggregate multi-probe probabilities into a stability estimate.
  4. Report stop reason and evidence sufficiency.

Adaptive stopping:
  Early stopping is ONLY applied when a stability threshold is explicitly
  configured AND has been validated on held-out data.

  If thresholds are null (not yet validated), the engine runs ALL probes
  deterministically and reports threshold_not_configured.

  Do NOT invent stopping thresholds. The engine architecture is correct;
  the thresholds are a scientific question requiring validation data.

Stop reasons:
  all_probes_complete       : all probes ran, no early stopping configured
  threshold_not_configured  : stopping thresholds are null (default)
  early_stop_stable         : (future) stopped early because stability met threshold
  early_stop_unstable       : (future) stopped early because instability confirmed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .probes import ProbeConfig, ProbeResult, ProbeRunner, BUILTIN_PROBES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StressTestConfig:
    """Configuration for one stress-test run."""
    probes: list[ProbeConfig] = field(default_factory=lambda: list(BUILTIN_PROBES))

    # Stability threshold — must be set from validated evidence, NOT invented.
    # None = threshold not configured; run all probes.
    stability_threshold: Optional[float] = None

    # Minimum probes to run before considering early stopping.
    min_probes_before_stop: int = 2


@dataclass
class StressTestResult:
    """
    Aggregated result of a stress-test run on one image.

    Probabilities are only aggregated from SUCCESSFUL probes.
    Failed probes are counted and reported but excluded from statistics.
    """
    # Original image inference
    baseline_raw_probability: float
    baseline_calibrated_probability: Optional[float]
    baseline_calibration_status: str

    # Probe summary
    probes_configured: int
    probes_run: int
    probes_successful: int
    probes_failed: int

    # Aggregated statistics (from successful probes only, not including baseline)
    probe_probabilities: list[float] = field(default_factory=list)
    mean_probability: Optional[float] = None
    std_probability: Optional[float] = None
    stability: Optional[float] = None  # 1 - std; None if insufficient data

    # Final decision
    final_prediction: str = "unknown"  # "real" | "synthetic" | "unknown"
    final_probability: Optional[float] = None  # calibrated if available, else raw

    # Stop reason
    stop_reason: str = "all_probes_complete"
    threshold_configured: bool = False
    evidence_sufficient: bool = False

    # Individual probe results
    probe_results: list[ProbeResult] = field(default_factory=list)

    # Timing
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "baseline_raw_probability":        self.baseline_raw_probability,
            "baseline_calibrated_probability": self.baseline_calibrated_probability,
            "baseline_calibration_status":     self.baseline_calibration_status,
            "probes_configured":               self.probes_configured,
            "probes_run":                      self.probes_run,
            "probes_successful":               self.probes_successful,
            "probes_failed":                   self.probes_failed,
            "probe_probabilities":             self.probe_probabilities,
            "mean_probability":                self.mean_probability,
            "std_probability":                 self.std_probability,
            "stability":                       self.stability,
            "final_prediction":                self.final_prediction,
            "final_probability":               self.final_probability,
            "stop_reason":                     self.stop_reason,
            "threshold_configured":            self.threshold_configured,
            "evidence_sufficient":             self.evidence_sufficient,
            "total_time_s":                    self.total_time_s,
            "probe_results":                   [p.to_dict() for p in self.probe_results],
        }


# ---------------------------------------------------------------------------
# Adaptive stress-testing engine
# ---------------------------------------------------------------------------

class StressTestEngine:
    """
    Adaptive stress-testing engine for SignalScope.

    Runs baseline + probe inferences and aggregates results.
    Early stopping is a planned feature; without validated thresholds,
    the engine runs all probes deterministically.

    Parameters
    ----------
    probe_runner:
        Configured ProbeRunner with loaded model and calibration.
    config:
        StressTestConfig. Defaults to running all BUILTIN_PROBES.
    """

    def __init__(
        self,
        probe_runner: ProbeRunner,
        config: Optional[StressTestConfig] = None,
    ) -> None:
        self.runner = probe_runner
        self.config = config or StressTestConfig()

    def run(self, image) -> StressTestResult:
        """
        Run full stress test on a PIL image.

        Parameters
        ----------
        image:
            PIL.Image.Image — the original, unmodified image.

        Returns
        -------
        StressTestResult
        """
        from PIL import Image as PILImage
        t0 = time.perf_counter()

        # ── Baseline inference ────────────────────────────────────────────
        baseline_raw = self.runner._infer(image)
        baseline_cal, baseline_cal_st = self.runner._apply_calibration(baseline_raw)

        # ── Run all probes ────────────────────────────────────────────────
        probe_results: list[ProbeResult] = []
        successful_probs: list[float] = []

        threshold_configured = self.config.stability_threshold is not None

        for probe in self.config.probes:
            result = self.runner.run_probe(image, probe)
            probe_results.append(result)

            if result.success and result.raw_probability is not None:
                # Use calibrated probability for aggregation if available
                p = result.calibrated_probability
                if p is None:
                    p = result.raw_probability
                successful_probs.append(p)

        # ── Aggregate statistics ──────────────────────────────────────────
        n_success = len(successful_probs)
        n_failed  = len(probe_results) - n_success

        mean_p = float(np.mean(successful_probs)) if successful_probs else None
        std_p  = float(np.std(successful_probs, ddof=1)) if len(successful_probs) >= 2 else None
        # stability = 1 - std; higher = more consistent
        stability = (1.0 - min(std_p, 1.0)) if std_p is not None else None

        # ── Evidence sufficiency ──────────────────────────────────────────
        # Require at least 2 successful probes for any stability claim
        evidence_sufficient = n_success >= 2

        # ── Stop reason ───────────────────────────────────────────────────
        if not threshold_configured:
            stop_reason = "threshold_not_configured"
        else:
            stop_reason = "all_probes_complete"

        # ── Final prediction ──────────────────────────────────────────────
        # Use calibrated baseline probability as the primary output
        final_prob = baseline_cal if baseline_cal is not None else baseline_raw
        final_pred = "synthetic" if final_prob >= 0.5 else "real"

        return StressTestResult(
            baseline_raw_probability=round(baseline_raw, 6),
            baseline_calibrated_probability=round(baseline_cal, 6) if baseline_cal is not None else None,
            baseline_calibration_status=baseline_cal_st,
            probes_configured=len(self.config.probes),
            probes_run=len(probe_results),
            probes_successful=n_success,
            probes_failed=n_failed,
            probe_probabilities=[round(p, 6) for p in successful_probs],
            mean_probability=round(mean_p, 6) if mean_p is not None else None,
            std_probability=round(std_p, 6) if std_p is not None else None,
            stability=round(stability, 6) if stability is not None else None,
            final_prediction=final_pred,
            final_probability=round(final_prob, 6),
            stop_reason=stop_reason,
            threshold_configured=threshold_configured,
            evidence_sufficient=evidence_sufficient,
            probe_results=probe_results,
            total_time_s=round(time.perf_counter() - t0, 3),
        )
