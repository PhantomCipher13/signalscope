from typing import Optional
from .status import ReliabilityStatus, StabilityStatus, ReliabilityResult
from .aggregator import ProbabilityAggregator

class ReliabilityEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.minimum_observations = self.config.get('minimum_observations', None)
        self.stability_threshold = self.config.get('stability_threshold', None)
        self.n_versions = self.config.get('n_versions', None)
        self.aggregator = ProbabilityAggregator()
        
    def analyze_single(self, baseline_probability: float, calibration_status: str = 'not_calibrated', calibrated_probability: float | None = None) -> ReliabilityResult:
        threshold_configured = self.minimum_observations is not None and self.stability_threshold is not None
        return ReliabilityResult(
            baseline_probability=baseline_probability,
            calibrated_probability=calibrated_probability,
            calibration_status=calibration_status,
            observation_count=1,
            probabilities=[baseline_probability],
            mean_probability=baseline_probability,
            standard_deviation=None,
            stability=None,
            stability_status=StabilityStatus.INSUFFICIENT_EVIDENCE,
            reliability_status=ReliabilityStatus.INSUFFICIENT_EVIDENCE,
            reliability_note="Only one observation. Cannot assess transformation stability.",
            threshold_configured=threshold_configured
        )
        
    def analyze_probes(self, baseline_probability: float, probe_probabilities: list[float], calibration_status: str = 'not_calibrated', calibrated_probability: float | None = None) -> ReliabilityResult:
        all_probs = [baseline_probability] + probe_probabilities
        agg_result = self.aggregator.aggregate(all_probs, self.minimum_observations)
        
        threshold_configured = self.minimum_observations is not None and self.stability_threshold is not None
        
        if not threshold_configured:
            reliability_status = ReliabilityStatus.THRESHOLD_NOT_CONFIGURED
        elif agg_result["observation_count"] < self.minimum_observations:
            reliability_status = ReliabilityStatus.INSUFFICIENT_EVIDENCE
        elif agg_result["standard_deviation"] is not None and agg_result["standard_deviation"] < self.stability_threshold:
            reliability_status = ReliabilityStatus.STABLE
        else:
            reliability_status = ReliabilityStatus.UNSTABLE
                
        note = agg_result["note"]
        if not threshold_configured:
            note = "Thresholds for stability analysis are not configured."
            
        return ReliabilityResult(
            baseline_probability=baseline_probability,
            calibrated_probability=calibrated_probability,
            calibration_status=calibration_status,
            observation_count=agg_result["observation_count"],
            probabilities=all_probs,
            mean_probability=agg_result["mean_probability"],
            standard_deviation=agg_result["standard_deviation"],
            stability=agg_result["stability"],
            stability_status=agg_result["stability_status"],
            reliability_status=reliability_status,
            reliability_note=note,
            threshold_configured=threshold_configured
        )

def format_result(result: ReliabilityResult) -> str:
    parts = [
        f"Reliability Status: {result.reliability_status.value}",
        f"Observation Count: {result.observation_count}",
        f"Note: {result.reliability_note}"
    ]
    if result.stability is not None:
        parts.append(f"Stability: {result.stability:.4f}")
    if result.mean_probability is not None:
        parts.append(f"Mean Probability: {result.mean_probability:.4f}")
    if result.standard_deviation is not None:
        parts.append(f"Standard Deviation: {result.standard_deviation:.4f}")
        
    return "\n".join(parts)
