import statistics
from .status import StabilityStatus

class ProbabilityAggregator:
    """
    Computes summary statistics for a sequence of probability observations.
    
    Scientific note:
    stability = 1 - std is a simple heuristic, not a formally validated measure.
    std with ddof=1 requires n>=2. For n=1, std=0 is mathematically true but
    does NOT mean the system is stable; it means there is only one observation.
    minimum_observations should be set via config, not hardcoded.
    """
    
    def aggregate(self, probs: list[float], minimum_observations: int | None) -> dict:
        n = len(probs)
        mean_prob = sum(probs) / n if n > 0 else None
        
        std_dev = None
        if n >= 2:
            std_dev = statistics.stdev(probs)
            
        stability = None
        stability_status = StabilityStatus.UNAVAILABLE
        note = ""
        
        if minimum_observations is None:
            note = "minimum_observations threshold not configured."
        elif n < minimum_observations:
            stability_status = StabilityStatus.INSUFFICIENT_EVIDENCE
            note = f"Insufficient evidence: {n} observations (minimum required: {minimum_observations})."
        else:
            if std_dev is not None:
                stability = max(0.0, min(1.0, 1.0 - std_dev))
                stability_status = StabilityStatus.COMPUTED
                note = "Stability computed successfully."
            else:
                stability_status = StabilityStatus.INSUFFICIENT_EVIDENCE
                note = "Insufficient observations to compute standard deviation."
                
        return {
            "observation_count": n,
            "mean_probability": mean_prob,
            "standard_deviation": std_dev,
            "stability": stability,
            "stability_status": stability_status,
            "note": note
        }
