"""
SignalScope — src/robustness/probes.py
========================================
Deterministic image transformation probes for robustness evaluation.

Design principles:
  1. Each probe is derived from the ORIGINAL image, not chained.
  2. Probes are deterministic — same input always gives same output.
  3. Failed probes are reported explicitly, never silently skipped.
  4. Probabilities are never fabricated for failed probes.
  5. The probe runner does not modify the original image.

Probe types implemented:
  - jpeg_qN   : JPEG re-encode at quality N (round-trip through JPEG codec)
  - resize_pct: Bilinear resize to P% of original dimensions, then back to original

These probes test resilience to common real-world image transformations.
They are NOT adversarial perturbations.

SCIENTIFIC NOTE:
  Probe results measure transformation sensitivity.
  They do NOT measure generalisation to unseen generators.
  A model may maintain high probe accuracy while failing entirely on
  distributions not seen during training.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Specification for a single probe transformation."""
    name: str          # Unique identifier, e.g. "jpeg_q70"
    transform_fn: Callable[[Image.Image], Image.Image]
    description: str   # Human-readable
    parameters: dict   # Logged with results for reproducibility


@dataclass
class ProbeResult:
    """Result of running one probe transformation through the detector."""
    probe_name: str
    description: str
    parameters: dict

    success: bool
    error: Optional[str] = None

    # Raw model output — only set on success
    raw_probability: Optional[float] = None
    # Calibrated probability — set if calibration was available and applied
    calibrated_probability: Optional[float] = None
    calibration_status: str = "not_calibrated"

    processing_time_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "probe_name":             self.probe_name,
            "description":            self.description,
            "parameters":             self.parameters,
            "success":                self.success,
            "error":                  self.error,
            "raw_probability":        self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "calibration_status":     self.calibration_status,
            "processing_time_s":      self.processing_time_s,
        }


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def _jpeg_encode_decode(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip an image through JPEG codec at the specified quality."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _resize_and_restore(image: Image.Image, scale_pct: int) -> Image.Image:
    """Resize to scale_pct% then back to original size (bilinear)."""
    w, h = image.size
    small_w = max(1, int(w * scale_pct / 100))
    small_h = max(1, int(h * scale_pct / 100))
    small = image.resize((small_w, small_h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Built-in probe catalogue (deterministic order)
# ---------------------------------------------------------------------------

BUILTIN_PROBES: list[ProbeConfig] = [
    ProbeConfig(
        name="jpeg_q90",
        transform_fn=lambda img: _jpeg_encode_decode(img, quality=90),
        description="JPEG re-encode at quality 90 (mild compression)",
        parameters={"format": "JPEG", "quality": 90},
    ),
    ProbeConfig(
        name="jpeg_q70",
        transform_fn=lambda img: _jpeg_encode_decode(img, quality=70),
        description="JPEG re-encode at quality 70 (moderate compression)",
        parameters={"format": "JPEG", "quality": 70},
    ),
    ProbeConfig(
        name="jpeg_q50",
        transform_fn=lambda img: _jpeg_encode_decode(img, quality=50),
        description="JPEG re-encode at quality 50 (strong compression)",
        parameters={"format": "JPEG", "quality": 50},
    ),
    ProbeConfig(
        name="resize_75pct",
        transform_fn=lambda img: _resize_and_restore(img, scale_pct=75),
        description="Resize to 75% then restore to original dimensions (bilinear)",
        parameters={"scale_pct": 75, "interpolation": "bilinear"},
    ),
    ProbeConfig(
        name="resize_50pct",
        transform_fn=lambda img: _resize_and_restore(img, scale_pct=50),
        description="Resize to 50% then restore to original dimensions (bilinear)",
        parameters={"scale_pct": 50, "interpolation": "bilinear"},
    ),
]


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

class ProbeRunner:
    """
    Runs deterministic image probes through a loaded model.

    Usage:
        runner = ProbeRunner(model, transform, device, calibration_temperature=T)
        results = runner.run(image, probes=BUILTIN_PROBES)
    """

    def __init__(
        self,
        model,                  # nn.Module, already on device and in eval mode
        val_transform,          # torchvision transform (same as used during training)
        device,                 # torch.device
        calibration_temperature: Optional[float] = None,
        calibration_status: str = "not_calibrated",
    ) -> None:
        self.model = model
        self.transform = val_transform
        self.device = device
        self.temperature = calibration_temperature
        self.calibration_status = calibration_status

    def _infer(self, image: Image.Image) -> float:
        """Run inference on a PIL image; return P(synthetic)."""
        import torch
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)
                return probs[0, 1].item()
            else:
                return torch.sigmoid(logits[0, 0]).item()

    def _apply_calibration(self, raw_prob: float) -> tuple[float, str]:
        """
        Apply temperature scaling if available.
        Returns (calibrated_probability, calibration_status).
        """
        import numpy as np
        if (
            self.calibration_status == "calibrated"
            and self.temperature is not None
            and self.temperature > 0
        ):
            try:
                _eps = 1e-7
                _p   = float(np.clip(raw_prob, _eps, 1.0 - _eps))
                raw_logit    = float(np.log(_p / (1.0 - _p)))
                scaled_logit = raw_logit / self.temperature
                cal_prob     = float(1.0 / (1.0 + np.exp(-scaled_logit)))
                return cal_prob, "calibrated"
            except Exception:
                return None, "calibration_error"
        return None, self.calibration_status

    def run_probe(self, image: Image.Image, probe: ProbeConfig) -> ProbeResult:
        """Run one probe. Returns ProbeResult with success/failure clearly recorded."""
        t0 = time.perf_counter()
        try:
            transformed = probe.transform_fn(image)
            raw_prob = self._infer(transformed)
            cal_prob, cal_st = self._apply_calibration(raw_prob)
            elapsed = time.perf_counter() - t0
            return ProbeResult(
                probe_name=probe.name,
                description=probe.description,
                parameters=probe.parameters,
                success=True,
                raw_probability=round(raw_prob, 6),
                calibrated_probability=round(cal_prob, 6) if cal_prob is not None else None,
                calibration_status=cal_st,
                processing_time_s=round(elapsed, 3),
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return ProbeResult(
                probe_name=probe.name,
                description=probe.description,
                parameters=probe.parameters,
                success=False,
                error=str(e),
                processing_time_s=round(elapsed, 3),
            )

    def run(
        self,
        image: Image.Image,
        probes: list[ProbeConfig] | None = None,
    ) -> list[ProbeResult]:
        """
        Run all specified probes in deterministic order.

        Parameters
        ----------
        image:
            Original PIL image. NOT mutated.
        probes:
            List of ProbeConfig to run. Defaults to BUILTIN_PROBES.

        Returns
        -------
        list[ProbeResult]
            One ProbeResult per probe, in the same order as `probes`.
        """
        if probes is None:
            probes = BUILTIN_PROBES
        return [self.run_probe(image, p) for p in probes]
