from enum import Enum
from dataclasses import dataclass, asdict

class ReliabilityStatus(str, Enum):
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
    STABLE = 'stable'
    UNSTABLE = 'unstable'
    UNAVAILABLE = 'unavailable'
    THRESHOLD_NOT_CONFIGURED = 'threshold_not_configured'

class StabilityStatus(str, Enum):
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
    COMPUTED = 'computed'
    UNAVAILABLE = 'unavailable'

@dataclass
class ReliabilityResult:
    baseline_probability: float
    calibrated_probability: float | None
    calibration_status: str
    observation_count: int
    probabilities: list[float]
    mean_probability: float | None
    standard_deviation: float | None
    stability: float | None
    stability_status: StabilityStatus
    reliability_status: ReliabilityStatus
    reliability_note: str
    threshold_configured: bool
    
    def to_dict(self) -> dict:
        return asdict(self)
