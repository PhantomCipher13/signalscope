import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional

class CalibrationStatus(str, Enum):
    NOT_CALIBRATED = 'not_calibrated'
    CALIBRATED = 'calibrated'
    CALIBRATION_UNAVAILABLE = 'calibration_unavailable'
    CALIBRATION_FALLBACK = 'calibration_fallback'

@dataclass
class CalibrationResult:
    temperature: Optional[float]
    status: CalibrationStatus
    fitted_on: Optional[str]
    n_samples_fitted: Optional[int]
    ece_before: Optional[float]
    ece_after: Optional[float]
    brier_before: Optional[float]
    brier_after: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value if isinstance(self.status, Enum) else self.status
        return d
