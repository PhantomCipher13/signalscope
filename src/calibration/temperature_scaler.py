import json
import logging
import numpy as np
from typing import Optional, Tuple

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .status import CalibrationStatus, CalibrationResult
from .metrics import compute_ece, compute_brier

logger = logging.getLogger(__name__)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    eps = 1e-15
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

class TemperatureScaler:
    """
    Temperature Scaler for calibrating model predictions.
    Temperature must ONLY be fit on validation data.
    """
    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False
        self.calibration_result: Optional[CalibrationResult] = None
        self._fitted_on_split: Optional[str] = None

    def fit(self, logits_or_probs: np.ndarray, labels: np.ndarray, split: str = 'validation') -> CalibrationResult:
        if split != 'validation':
            logger.warning("Temperature must ONLY be fit on validation data.")
            
        logits_or_probs = np.asarray(logits_or_probs)
        labels = np.asarray(labels)

        if len(logits_or_probs) == 0:
            res = CalibrationResult(
                temperature=None,
                status=CalibrationStatus.CALIBRATION_UNAVAILABLE,
                fitted_on=split,
                n_samples_fitted=0,
                ece_before=None, ece_after=None,
                brier_before=None, brier_after=None
            )
            self.calibration_result = res
            return res

        # One observation cannot determine stability
        if len(logits_or_probs) == 1:
            res = CalibrationResult(
                temperature=None,
                status=CalibrationStatus.CALIBRATION_UNAVAILABLE,
                fitted_on=split,
                n_samples_fitted=1,
                ece_before=None, ece_after=None,
                brier_before=None, brier_after=None
            )
            self.calibration_result = res
            return res

        # Auto-detect logits vs probs
        is_probs = np.all((logits_or_probs >= 0) & (logits_or_probs <= 1))
        if is_probs:
            logits = inv_sigmoid(logits_or_probs)
            probs = logits_or_probs
        else:
            logits = logits_or_probs
            probs = sigmoid(logits)

        ece_before = compute_ece(labels, probs)
        brier_before = compute_brier(labels, probs)

        def nll(t):
            t = max(t, 1e-5)
            scaled_logits = logits / t
            scaled_probs = sigmoid(scaled_logits)
            eps = 1e-15
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
            return -np.sum(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))

        try:
            if HAS_SCIPY:
                res = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
                if res.success:
                    self.temperature = res.x
                    self.is_fitted = True
                    status = CalibrationStatus.CALIBRATED
                else:
                    raise ValueError("scipy optimization failed")
            else:
                # Fallback to grid search
                t_grid = np.linspace(0.1, 5.0, 50)
                losses = [nll(t) for t in t_grid]
                best_idx = np.argmin(losses)
                self.temperature = t_grid[best_idx]
                self.is_fitted = True
                status = CalibrationStatus.CALIBRATION_FALLBACK

            scaled_probs = sigmoid(logits / self.temperature)
            ece_after = compute_ece(labels, scaled_probs)
            brier_after = compute_brier(labels, scaled_probs)

            self._fitted_on_split = split
            
            self.calibration_result = CalibrationResult(
                temperature=float(self.temperature),
                status=status,
                fitted_on=split,
                n_samples_fitted=len(labels),
                ece_before=ece_before,
                ece_after=ece_after,
                brier_before=brier_before,
                brier_after=brier_after
            )

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.calibration_result = CalibrationResult(
                temperature=None,
                status=CalibrationStatus.CALIBRATION_UNAVAILABLE,
                fitted_on=split,
                n_samples_fitted=len(labels),
                ece_before=ece_before, ece_after=None,
                brier_before=brier_before, brier_after=None
            )

        return self.calibration_result

    def transform(self, logits_or_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("TemperatureScaler is not fitted. Returning input unchanged.")
            return np.asarray(logits_or_probs)
            
        if self.temperature == 1.0 and not self.is_fitted:
            return np.asarray(logits_or_probs)

        logits_or_probs = np.asarray(logits_or_probs)
        is_probs = np.all((logits_or_probs >= 0) & (logits_or_probs <= 1))
        
        if is_probs:
            logits = inv_sigmoid(logits_or_probs)
        else:
            logits = logits_or_probs
            
        scaled_logits = logits / self.temperature
        return sigmoid(scaled_logits)

    def transform_with_status(self, logits_or_probs: np.ndarray) -> Tuple[np.ndarray, CalibrationStatus]:
        probs = self.transform(logits_or_probs)
        return probs, self.calibration_status

    def save(self, path: str) -> None:
        data = {
            'temperature': float(self.temperature),
            'is_fitted': self.is_fitted,
            '_fitted_on_split': self._fitted_on_split,
            'calibration_result': self.calibration_result.to_dict() if self.calibration_result else None
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TemperatureScaler':
        with open(path, 'r') as f:
            data = json.load(f)
            
        scaler = cls()
        t = data.get('temperature', 1.0)
        if t <= 0:
            raise ValueError("Temperature must be > 0")
            
        scaler.temperature = t
        scaler.is_fitted = data.get('is_fitted', False)
        scaler._fitted_on_split = data.get('_fitted_on_split')
        
        cres = data.get('calibration_result')
        if cres:
            scaler.calibration_result = CalibrationResult(
                temperature=cres.get('temperature'),
                status=CalibrationStatus(cres.get('status')),
                fitted_on=cres.get('fitted_on'),
                n_samples_fitted=cres.get('n_samples_fitted'),
                ece_before=cres.get('ece_before'),
                ece_after=cres.get('ece_after'),
                brier_before=cres.get('brier_before'),
                brier_after=cres.get('brier_after')
            )
            
        return scaler

    @property
    def calibration_status(self) -> CalibrationStatus:
        if self.calibration_result:
            return self.calibration_result.status
        if self.is_fitted:
            return CalibrationStatus.CALIBRATED
        return CalibrationStatus.NOT_CALIBRATED
