"""
Tests for src/gradcam.py and src/provenance.py

Coverage (Grad-CAM):
- GradCAMResult to_dict() has required keys
- heatmap_bytes() returns bytes on success
- overlay_bytes() returns bytes on success
- Colormap produces RGB image of correct size
- _find_target_layer finds Conv2d in a simple model
- GradCAM.generate() returns ok status with mocked model
- GradCAM.generate() returns failed status when model raises
- GradCAM does not modify the input image

Coverage (Provenance):
- ProvenanceAnalyser.analyse() always returns a ProvenanceResult
- EXIF absent status for PNG without EXIF
- Summary includes non-judgmental text about missing metadata
- C2PA status is c2pa_library_unavailable when library not installed
- ExifData.to_dict() has required keys
- C2PAData.to_dict() has required keys
- ProvenanceResult.to_dict() has required keys
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.gradcam import (
    GradCAM,
    GradCAMResult,
    GRADCAM_STATUS_OK,
    GRADCAM_STATUS_FAILED,
    GRADCAM_STATUS_UNSUPPORTED,
    _find_target_layer,
)
from src.provenance import (
    ProvenanceAnalyser,
    ExifData,
    C2PAData,
    ProvenanceResult,
    PROVENANCE_STATUS_AVAILABLE,
    PROVENANCE_STATUS_ABSENT,
    PROVENANCE_STATUS_UNAVAILABLE,
    C2PA_STATUS_UNSUPPORTED,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rgb_image_64():
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


@pytest.fixture
def simple_cnn():
    """A tiny 2-class CNN with a Conv2d layer for testing Grad-CAM hooks."""
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
            self.relu  = nn.ReLU()
            self.pool  = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(4, 2)
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    model = TinyCNN()
    model.eval()
    return model


def _identity_transform(img):
    """Simple transform: PIL → tensor."""
    import torchvision.transforms.functional as F
    return F.to_tensor(img)


# ── GradCAMResult tests ───────────────────────────────────────────────────────

class TestGradCAMResult:

    def test_to_dict_has_required_keys(self):
        r = GradCAMResult(status=GRADCAM_STATUS_OK, target_class=1)
        d = r.to_dict()
        assert "status" in d
        assert "target_layer_name" in d
        assert "target_class" in d
        assert "heatmap_available" in d
        assert "overlay_available" in d
        assert "error" in d

    def test_heatmap_bytes_none_when_no_heatmap(self):
        r = GradCAMResult(status=GRADCAM_STATUS_FAILED)
        assert r.heatmap_bytes() is None

    def test_overlay_bytes_none_when_no_overlay(self):
        r = GradCAMResult(status=GRADCAM_STATUS_FAILED)
        assert r.overlay_bytes() is None

    def test_heatmap_bytes_returns_bytes(self):
        heatmap = Image.new("RGB", (32, 32), color=(255, 0, 0))
        r = GradCAMResult(status=GRADCAM_STATUS_OK, heatmap_pil=heatmap)
        b = r.heatmap_bytes(fmt="PNG")
        assert isinstance(b, bytes)
        assert len(b) > 0

    def test_overlay_bytes_returns_bytes(self):
        overlay = Image.new("RGB", (32, 32), color=(128, 128, 0))
        r = GradCAMResult(status=GRADCAM_STATUS_OK, overlay_pil=overlay)
        b = r.overlay_bytes(fmt="PNG")
        assert isinstance(b, bytes)
        assert len(b) > 0

    def test_heatmap_available_true_when_present(self):
        heatmap = Image.new("RGB", (32, 32))
        r = GradCAMResult(status=GRADCAM_STATUS_OK, heatmap_pil=heatmap)
        assert r.to_dict()["heatmap_available"] is True

    def test_heatmap_available_false_when_absent(self):
        r = GradCAMResult(status=GRADCAM_STATUS_FAILED)
        assert r.to_dict()["heatmap_available"] is False


# ── _find_target_layer tests ──────────────────────────────────────────────────

class TestFindTargetLayer:

    def test_finds_conv2d_in_simple_cnn(self, simple_cnn):
        layer, name = _find_target_layer(simple_cnn)
        assert layer is not None
        assert isinstance(layer, nn.Conv2d)

    def test_returns_layer_name_string(self, simple_cnn):
        layer, name = _find_target_layer(simple_cnn)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_no_conv2d_returns_none(self):
        class NoConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
            def forward(self, x): return self.fc(x)
        layer, name = _find_target_layer(NoConvModel())
        assert layer is None


# ── GradCAM.generate() tests ──────────────────────────────────────────────────

class TestGradCAMGenerate:

    def test_generate_ok_with_simple_cnn(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        assert result.status == GRADCAM_STATUS_OK
        assert result.heatmap_pil is not None
        assert result.overlay_pil is not None

    def test_overlay_same_size_as_input(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        assert result.status == GRADCAM_STATUS_OK
        assert result.overlay_pil.size == rgb_image_64.size

    def test_heatmap_same_size_as_input(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        assert result.status == GRADCAM_STATUS_OK
        assert result.heatmap_pil.size == rgb_image_64.size

    def test_target_class_recorded(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        assert result.target_class == 1

    def test_original_image_not_modified(self, simple_cnn, rgb_image_64):
        import numpy as np
        orig_arr = np.array(rgb_image_64.copy())
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        gc.generate(rgb_image_64, transform=_identity_transform)
        after_arr = np.array(rgb_image_64)
        assert (orig_arr == after_arr).all()

    def test_hooks_removed_after_generate(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        gc.generate(rgb_image_64, transform=_identity_transform)
        assert len(gc._hooks) == 0

    def test_generate_failed_when_model_raises(self, rgb_image_64):
        bad_model = MagicMock()
        bad_model.named_modules = MagicMock(return_value=iter([
            ("conv1", nn.Conv2d(3, 4, 3))
        ]))
        bad_model.__call__ = MagicMock(side_effect=RuntimeError("OOM"))
        bad_model.eval     = MagicMock()
        bad_model.zero_grad= MagicMock()

        gc = GradCAM(model=bad_model, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        assert result.status == GRADCAM_STATUS_FAILED
        assert result.error is not None

    def test_generate_result_to_dict_has_required_keys(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"), target_class=1)
        result = gc.generate(rgb_image_64, transform=_identity_transform)
        d = result.to_dict()
        assert "status" in d
        assert "target_layer_name" in d
        assert "target_class" in d


# ── Colormap test ─────────────────────────────────────────────────────────────

class TestColormap:

    def test_colormap_returns_rgb_image(self, simple_cnn, rgb_image_64):
        gc = GradCAM(model=simple_cnn, device=torch.device("cpu"))
        import numpy as np
        arr = np.zeros((64, 64), dtype=float)
        arr[32:, 32:] = 1.0  # non-trivial
        result = gc._apply_colormap(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (64, 64)


# ── ProvenanceAnalyser tests ──────────────────────────────────────────────────

class TestProvenanceAnalyser:

    def _save_png(self, tmp_path, name="test.png") -> Path:
        path = tmp_path / name
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        img.save(str(path), format="PNG")
        return path

    def _save_jpeg(self, tmp_path, name="test.jpg") -> Path:
        path = tmp_path / name
        img = Image.new("RGB", (64, 64), color=(100, 200, 150))
        img.save(str(path), format="JPEG")
        return path

    def test_returns_provenance_result(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        assert isinstance(result, ProvenanceResult)

    def test_result_has_exif_and_c2pa(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        assert hasattr(result, "exif")
        assert hasattr(result, "c2pa")

    def test_png_exif_absent_or_unavailable(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        # PNG without EXIF block → absent, or unsupported (format-dependent)
        assert result.exif.status in (
            PROVENANCE_STATUS_ABSENT,
            "unsupported_format",
            PROVENANCE_STATUS_UNAVAILABLE,
        )

    def test_jpeg_exif_status_is_not_error(self, tmp_path):
        path = self._save_jpeg(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        # JPEG should parse cleanly (absent or available, never crash)
        assert result.exif.status in (
            PROVENANCE_STATUS_ABSENT,
            PROVENANCE_STATUS_AVAILABLE,
            PROVENANCE_STATUS_UNAVAILABLE,
        )

    def test_c2pa_status_unavailable_without_library(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        # c2pa library not installed → unsupported
        assert result.c2pa.status == C2PA_STATUS_UNSUPPORTED

    def test_summary_is_non_empty_string(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_does_not_claim_ai_from_absent_metadata(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        summary_lower = result.summary.lower()
        # Must NOT imply AI generation from missing metadata
        forbidden = ["ai generated", "synthetic", "fake", "manipulated", "evidence of"]
        for phrase in forbidden:
            assert phrase not in summary_lower, (
                f"Summary incorrectly implies AI from absent metadata: {result.summary!r}"
            )

    def test_exif_to_dict_has_required_keys(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        d = result.exif.to_dict()
        assert "status" in d
        assert "camera_make" in d
        assert "software" in d
        assert "gps_available" in d

    def test_c2pa_to_dict_has_required_keys(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        d = result.c2pa.to_dict()
        assert "status" in d
        assert "manifest_present" in d

    def test_provenance_to_dict_has_required_keys(self, tmp_path):
        path = self._save_png(tmp_path)
        result = ProvenanceAnalyser().analyse(path)
        d = result.to_dict()
        assert "exif" in d
        assert "c2pa" in d
        assert "summary" in d

    def test_nonexistent_file_does_not_raise(self):
        """analyse() must never raise — always returns a result."""
        result = ProvenanceAnalyser().analyse(Path("/nonexistent/image.jpg"))
        assert isinstance(result, ProvenanceResult)
        assert result.exif.status in (
            PROVENANCE_STATUS_UNAVAILABLE,
            "unsupported_format",
            PROVENANCE_STATUS_ABSENT,
        )
