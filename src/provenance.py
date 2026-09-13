"""
SignalScope — src/provenance.py
=================================
Image provenance and metadata analysis (EXIF / C2PA).

CRITICAL DESIGN CONSTRAINT:
    Missing or absent metadata MUST NEVER be interpreted as evidence
    that an image is AI-generated. Provenance is CONTEXTUAL ONLY.
    The detector score must not be adjusted based on metadata presence/absence.
    Metadata absence ≠ synthetic. Many real images lack EXIF.

Possible provenance states:
    - metadata_available   : EXIF data found and parsed
    - metadata_absent      : Image contains no EXIF data
    - c2pa_present         : C2PA Content Credentials manifest found
    - c2pa_absent          : No C2PA manifest detected
    - parsing_unavailable  : Library not installed / parsing failed
    - unsupported_format   : File format does not support EXIF

IMPORTANT:
    Never modify detector probability based on metadata.
    Never call absence of metadata a red flag.
    Return explicit unavailable states when data cannot be retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── Status codes ──────────────────────────────────────────────────────────────

PROVENANCE_STATUS_AVAILABLE   = "metadata_available"
PROVENANCE_STATUS_ABSENT      = "metadata_absent"
PROVENANCE_STATUS_UNAVAILABLE = "parsing_unavailable"
PROVENANCE_STATUS_UNSUPPORTED = "unsupported_format"

C2PA_STATUS_PRESENT   = "c2pa_present"
C2PA_STATUS_ABSENT    = "c2pa_absent"
C2PA_STATUS_UNKNOWN   = "c2pa_unknown"
C2PA_STATUS_UNSUPPORTED = "c2pa_library_unavailable"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExifData:
    """Parsed EXIF metadata. All fields may be None if not present."""
    status: str                   # one of PROVENANCE_STATUS_*

    camera_make:   Optional[str]  = None
    camera_model:  Optional[str]  = None
    software:      Optional[str]  = None
    datetime_original: Optional[str] = None
    gps_available: bool           = False
    image_width:   Optional[int]  = None
    image_height:  Optional[int]  = None
    color_space:   Optional[str]  = None
    orientation:   Optional[int]  = None
    flash:         Optional[str]  = None
    focal_length:  Optional[str]  = None
    iso_speed:     Optional[int]  = None
    exposure_time: Optional[str]  = None
    f_number:      Optional[str]  = None

    # Full raw tag dict (for completeness; may be large)
    raw_tags: dict = field(default_factory=dict)

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status":          self.status,
            "camera_make":     self.camera_make,
            "camera_model":    self.camera_model,
            "software":        self.software,
            "datetime_original": self.datetime_original,
            "gps_available":   self.gps_available,
            "image_width":     self.image_width,
            "image_height":    self.image_height,
            "color_space":     self.color_space,
            "orientation":     self.orientation,
            "flash":           self.flash,
            "focal_length":    self.focal_length,
            "iso_speed":       self.iso_speed,
            "exposure_time":   self.exposure_time,
            "f_number":        self.f_number,
            "error":           self.error,
            # Omit raw_tags from API response by default (can be large)
        }


@dataclass
class C2PAData:
    """C2PA Content Credentials result."""
    status: str   # one of C2PA_STATUS_*
    manifest_present: bool = False
    creator_tool: Optional[str] = None
    assertions: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status":           self.status,
            "manifest_present": self.manifest_present,
            "creator_tool":     self.creator_tool,
            "assertions":       self.assertions,
            "error":            self.error,
        }


@dataclass
class ProvenanceResult:
    """Combined provenance analysis result."""
    exif: ExifData
    c2pa: C2PAData
    # Summary: human-readable, non-judgmental
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "exif": self.exif.to_dict(),
            "c2pa": self.c2pa.to_dict(),
            "summary": self.summary,
        }


# ── Provenance analyser ────────────────────────────────────────────────────────

class ProvenanceAnalyser:
    """
    Extracts EXIF and C2PA metadata from image files.

    Design constraints:
      - Result is contextual information only.
      - Never adjusts detector probability.
      - Missing metadata → explicit 'metadata_absent' status.
      - Parse failures → explicit 'parsing_unavailable' status.
    """

    # EXIF tag IDs we care about (Pillow ExifTags.TAGS)
    _EXIF_TAGS = {
        271: "Make",
        272: "Model",
        305: "Software",
        36867: "DateTimeOriginal",
        256: "ImageWidth",
        257: "ImageLength",
        40961: "ColorSpace",
        274: "Orientation",
        37385: "Flash",
        37386: "FocalLength",
        34855: "ISOSpeedRatings",
        33434: "ExposureTime",
        33437: "FNumber",
        34853: "GPSInfo",
    }

    def analyse(self, image_path: Path) -> ProvenanceResult:
        """
        Extract all available provenance metadata.

        Parameters
        ----------
        image_path : Path
            Path to the image file.

        Returns
        -------
        ProvenanceResult
            Always returns a result. Never raises.
        """
        exif_data = self._extract_exif(image_path)
        c2pa_data = self._check_c2pa(image_path)
        summary   = self._build_summary(exif_data, c2pa_data)
        return ProvenanceResult(exif=exif_data, c2pa=c2pa_data, summary=summary)

    def _extract_exif(self, path: Path) -> ExifData:
        """Extract EXIF using Pillow. Returns ExifData with appropriate status."""
        supported_exif_formats = {".jpg", ".jpeg", ".tiff", ".tif", ".webp"}
        if path.suffix.lower() not in supported_exif_formats:
            return ExifData(
                status=PROVENANCE_STATUS_UNSUPPORTED,
                error=f"Format {path.suffix!r} does not typically carry EXIF data.",
            )

        try:
            from PIL import Image, ExifTags
            img = Image.open(path)

            # Try getexif() (Pillow ≥ 6.0)
            try:
                exif_obj = img.getexif()
                raw_tags = {
                    ExifTags.TAGS.get(k, str(k)): v
                    for k, v in exif_obj.items()
                    if isinstance(v, (str, int, float, bytes))
                }
            except AttributeError:
                raw_tags = {}

            if not raw_tags:
                return ExifData(status=PROVENANCE_STATUS_ABSENT)

            def tag(key: str) -> Any:
                return raw_tags.get(key)

            # Rational number helper
            def to_str(v: Any) -> Optional[str]:
                if v is None:
                    return None
                if hasattr(v, "numerator"):
                    return f"{v.numerator}/{v.denominator}"
                return str(v)

            return ExifData(
                status=PROVENANCE_STATUS_AVAILABLE,
                camera_make=tag("Make"),
                camera_model=tag("Model"),
                software=tag("Software"),
                datetime_original=tag("DateTimeOriginal"),
                gps_available="GPSInfo" in raw_tags,
                image_width=tag("ImageWidth"),
                image_height=tag("ImageLength"),
                orientation=tag("Orientation"),
                iso_speed=tag("ISOSpeedRatings"),
                exposure_time=to_str(tag("ExposureTime")),
                f_number=to_str(tag("FNumber")),
                focal_length=to_str(tag("FocalLength")),
                raw_tags=raw_tags,
            )

        except Exception as e:
            return ExifData(
                status=PROVENANCE_STATUS_UNAVAILABLE,
                error=str(e),
            )

    def _check_c2pa(self, path: Path) -> C2PAData:
        """
        Check for C2PA Content Credentials.
        The 'c2pa' Python library is not widely available as of 2024.
        Returns c2pa_library_unavailable if not installed.
        """
        try:
            import c2pa  # type: ignore
            # If library present, attempt basic manifest read
            try:
                manifest = c2pa.read_file(str(path))
                if manifest:
                    return C2PAData(
                        status=C2PA_STATUS_PRESENT,
                        manifest_present=True,
                        creator_tool=manifest.get("claim_generator", None),
                    )
                else:
                    return C2PAData(status=C2PA_STATUS_ABSENT)
            except Exception as e:
                return C2PAData(status=C2PA_STATUS_ABSENT, error=str(e))
        except ImportError:
            return C2PAData(
                status=C2PA_STATUS_UNSUPPORTED,
                error="c2pa library not installed. Install with: pip install c2pa-python",
            )

    @staticmethod
    def _build_summary(exif: ExifData, c2pa: C2PAData) -> str:
        """Build a non-judgmental human-readable summary."""
        parts = []

        if exif.status == PROVENANCE_STATUS_AVAILABLE:
            cam_parts = [
                x for x in [exif.camera_make, exif.camera_model] if x
            ]
            if cam_parts:
                parts.append(f"Camera: {' '.join(cam_parts)}")
            if exif.software:
                parts.append(f"Software: {exif.software}")
            if exif.datetime_original:
                parts.append(f"Taken: {exif.datetime_original}")
            if not parts:
                parts.append("EXIF data present but no camera/software tags.")
        elif exif.status == PROVENANCE_STATUS_ABSENT:
            parts.append("No EXIF metadata. (Common for web images; not indicative of AI generation.)")
        elif exif.status == PROVENANCE_STATUS_UNSUPPORTED:
            parts.append(f"EXIF not available for this file format.")
        else:
            parts.append(f"EXIF parsing unavailable: {exif.error}")

        if c2pa.status == C2PA_STATUS_PRESENT:
            parts.append("C2PA Content Credentials found.")
        elif c2pa.status == C2PA_STATUS_ABSENT:
            parts.append("No C2PA Content Credentials. (Absence is not evidence of AI generation.)")
        elif c2pa.status == C2PA_STATUS_UNSUPPORTED:
            parts.append("C2PA analysis unavailable (library not installed).")

        return " | ".join(parts) if parts else "No provenance data available."
