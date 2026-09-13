"""
Phase 1 unit tests — Preprocessing module.

Tests use tiny in-memory synthetic images for software validation only.
These images are NOT training data and generate no meaningful ML results.

Torch-dependent tests are skipped when torch DLLs are blocked by a
Windows Application Control policy. Run on Colab/unconstrained machine
for full coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from conftest import requires_torch  # noqa: E402
from src.preprocessing import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    load_image,
    is_valid_image,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_pil_image(width: int = 64, height: int = 64):
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_image(tmp_path: Path, fmt: str = "PNG", width: int = 64, height: int = 64) -> Path:
    img = _make_pil_image(width, height)
    ext = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "WEBP": ".webp"}[fmt]
    p = tmp_path / f"test{ext}"
    img.save(p, format=fmt)
    return p


# ── load_image tests (PIL only — no torch needed) ─────────────────────────

class TestLoadImage:
    def test_loads_png(self, tmp_path):
        p = _save_image(tmp_path, "PNG")
        img = load_image(p)
        assert img.mode == "RGB"

    def test_loads_jpeg(self, tmp_path):
        p = _save_image(tmp_path, "JPEG")
        img = load_image(p)
        assert img.mode == "RGB"

    def test_loads_bmp(self, tmp_path):
        p = _save_image(tmp_path, "BMP")
        img = load_image(p)
        assert img.mode == "RGB"

    def test_loads_webp(self, tmp_path):
        p = _save_image(tmp_path, "WEBP")
        img = load_image(p)
        assert img.mode == "RGB"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.png")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "test.tiff"
        p.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(p)

    def test_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"NOT AN IMAGE DATA")
        with pytest.raises(OSError):
            load_image(p)

    def test_is_valid_image_true(self, tmp_path):
        p = _save_image(tmp_path, "PNG")
        assert is_valid_image(p) is True

    def test_is_valid_image_false_corrupt(self, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"garbage")
        assert is_valid_image(p) is False

    def test_is_valid_image_false_missing(self, tmp_path):
        assert is_valid_image(tmp_path / "nonexistent.png") is False


# ── Transform tests (require torch) ───────────────────────────────────────

class TestTransforms:
    @requires_torch
    def test_val_transform_output_shape(self, tmp_path):
        import torch
        from src.preprocessing import build_val_transforms
        p = _save_image(tmp_path, "PNG", width=128, height=128)
        img = load_image(p)
        t = build_val_transforms(image_size=64)
        tensor = t(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    @requires_torch
    def test_val_transform_output_dtype(self, tmp_path):
        import torch
        from src.preprocessing import build_val_transforms
        p = _save_image(tmp_path, "PNG")
        img = load_image(p)
        t = build_val_transforms(image_size=32)
        tensor = t(img)
        assert tensor.dtype == torch.float32

    @requires_torch
    def test_val_transform_is_deterministic(self, tmp_path):
        import torch
        from src.preprocessing import build_val_transforms
        p = _save_image(tmp_path, "PNG", width=128, height=128)
        img = load_image(p)
        t = build_val_transforms(image_size=64)
        t1 = t(img)
        t2 = t(img)
        assert torch.allclose(t1, t2), "Val transform must be deterministic"

    @requires_torch
    def test_train_transform_output_shape(self, tmp_path):
        import torch
        from src.preprocessing import build_train_transforms
        p = _save_image(tmp_path, "PNG", width=128, height=128)
        img = load_image(p)
        t = build_train_transforms(image_size=64, jpeg_prob=0.0, resize_prob=0.0)
        tensor = t(img)
        assert tensor.shape == (3, 64, 64)

    @requires_torch
    def test_normalisation_mean_std(self, tmp_path):
        from PIL import Image
        from src.preprocessing import build_val_transforms
        img = Image.new("RGB", (96, 96), color=(255, 255, 255))
        t = build_val_transforms(image_size=64)
        tensor = t(img)
        assert tensor.mean().item() > 0
