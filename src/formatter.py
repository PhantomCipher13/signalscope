"""
SignalScope — src/formatter.py
================================
API response formatting and output structure.

Phase 0: Stub only — safe for import, no formatting logic implemented.

Phase 1+ will implement:
- Structured response assembly from detector, calibration,
  reliability, Grad-CAM, and provenance outputs.
- Calibrated probability → human-readable likelihood label
  ("Likely Real" / "Likely AI-Generated" / "Uncertain").
- Confidence margin computation.
- Reliability summary fields.
- Provenance context fields (contextual only).
- JSON-serialisable Pydantic response model.

Results are always expressed as likelihood, never absolute certainty.
The formatter must never fabricate or hallucinate evidence.

Do NOT add formatting logic here until Phase 1 is approved.
"""

from __future__ import annotations


class ResponseFormatter:
    """Placeholder for structured API response assembly.

    Phase 1+ will implement result aggregation and the Pydantic
    response model for the /predict endpoint.
    """

    def __init__(self) -> None:
        # Phase 1+: initialise label thresholds from config.
        pass
