"""
src/schemas.py
===============
Pydantic response models for the SignalScope API.

Represents the full analysis output including:
  - baseline probability (raw model output)
  - calibrated probability (when calibration is available)
  - calibration status (explicit — never silently uncalibrated)
  - reliability analysis (observation count, std, stability, status)

Future phases will extend these models with:
  - Grad-CAM attribution data (Phase 4)
  - Provenance / C2PA / EXIF metadata (Phase 5)

SCIENTIFIC INTEGRITY:
  - calibrated_probability is None when calibration is not available.
  - reliability.stability is None when only one observation exists.
  - All fields that are not yet computed are explicitly None.
  - No field is fabricated or defaulted to a misleading value.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Calibration status (mirrors src/calibration/status.py) ───────────────────

class CalibrationStatusEnum(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATED = "calibrated"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    CALIBRATION_FALLBACK = "calibration_fallback"


# ── Reliability status (mirrors src/reliability/status.py) ───────────────────

class ReliabilityStatusEnum(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STABLE = "stable"
    UNSTABLE = "unstable"
    UNAVAILABLE = "unavailable"
    THRESHOLD_NOT_CONFIGURED = "threshold_not_configured"


class StabilityStatusEnum(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    COMPUTED = "computed"
    UNAVAILABLE = "unavailable"


# ── Nested models ─────────────────────────────────────────────────────────────

class ReliabilitySchema(BaseModel):
    """Reliability analysis result for a single analysis.

    stability is None when only one observation exists (insufficient evidence).
    DO NOT return stability=1.0 for a single observation.
    """
    observation_count: int = Field(
        description="Number of probability observations (1 = only baseline, no probes yet)."
    )
    mean_probability: Optional[float] = Field(
        default=None,
        description="Mean probability across all observations.",
    )
    standard_deviation: Optional[float] = Field(
        default=None,
        description="Std of probabilities (None when observation_count < 2).",
    )
    stability: Optional[float] = Field(
        default=None,
        description=(
            "Stability score = 1 - std. None when insufficient observations. "
            "DO NOT interpret None as stable — it means insufficient_evidence."
        ),
    )
    stability_status: StabilityStatusEnum = Field(
        description="Whether stability was computable for this analysis."
    )
    status: ReliabilityStatusEnum = Field(
        description="Overall reliability classification."
    )
    note: str = Field(
        description="Human-readable explanation of the reliability status."
    )
    threshold_configured: bool = Field(
        description="Whether reliability thresholds have been configured from validation data."
    )


class CalibrationDetailSchema(BaseModel):
    """Optional calibration metadata (when calibration is available)."""
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature scaling factor T. None if not yet fitted.",
    )
    ece_before: Optional[float] = Field(
        default=None, description="ECE before calibration."
    )
    ece_after: Optional[float] = Field(
        default=None, description="ECE after calibration."
    )
    brier_before: Optional[float] = Field(
        default=None, description="Brier score before calibration."
    )
    brier_after: Optional[float] = Field(
        default=None, description="Brier score after calibration."
    )


# ── Primary response model ─────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """
    Full analysis response from SignalScope.

    Represents the system's assessment of whether an image is real or synthetic.
    Never expresses absolute certainty — only probabilistic likelihood.

    Phase 3 additions:
      - calibrated_probability (None if calibration not available)
      - calibration_status (always explicit)
      - reliability (always present, stability may be None)
    """
    # ── Prediction ──────────────────────────────────────────
    prediction: str = Field(
        description="Predicted class: 'real' or 'synthetic'."
    )
    label: str = Field(
        description="Human-readable likelihood: 'Likely Real', 'Likely AI-Generated', 'Uncertain'."
    )

    # ── Probabilities ───────────────────────────────────────
    baseline_probability: float = Field(
        ge=0.0, le=1.0,
        description="Raw model synthetic probability. NOT calibrated.",
    )
    calibrated_probability: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "Temperature-scaled probability. None when calibration is not available. "
            "Never silently substituted with baseline_probability."
        ),
    )
    calibration_status: CalibrationStatusEnum = Field(
        description="Explicit calibration state. Never silently claims calibrated."
    )

    # ── Reliability ─────────────────────────────────────────
    reliability: ReliabilitySchema = Field(
        description="Reliability analysis results."
    )

    # ── Optional details ────────────────────────────────────
    calibration_detail: Optional[CalibrationDetailSchema] = Field(
        default=None,
        description="Calibration metadata (only when calibration is available).",
    )
    image_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the input image.",
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Total processing time in milliseconds.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Model version used for this analysis.",
    )

    model_config = {"use_enum_values": True}


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    calibration_loaded: bool
    version: str
    schema_version: int


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: Optional[str] = None
    code: str = "internal_error"
