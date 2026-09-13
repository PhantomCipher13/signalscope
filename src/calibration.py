"""
SignalScope — src/calibration.py
==================================
Probability calibration via temperature scaling.

Phase 0: Stub only — safe for import, no calibration logic implemented.

Phase 1 will implement:
- Temperature scaling (a single learned scalar T).
- Fitting T on the validation split ONLY — never on test data.
- Freezing T before any unseen-test evaluation.
- ECE (Expected Calibration Error) and Brier score measurement
  (these are measured results, not performance targets).
- Persistence of T to configs/model.yaml after fitting.

IMPORTANT: The unseen test set must NEVER be used for calibration,
model selection, or threshold tuning.

Do NOT add calibration logic here until Phase 1 is approved.
"""

from __future__ import annotations


class TemperatureScaler:
    """Placeholder for temperature-scaling calibration.

    Phase 1 will implement fitting on validation data, persistence of
    the temperature parameter, and calibrated probability output.

    Temperature T=1.0 means no scaling (identity).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        # Phase 1: load from config; fit on validation data.
        self.temperature: float = temperature
