"""
Phase 1 unit tests — Detector module.

All detector tests require torch. They are automatically skipped on
machines where a Windows Application Control policy blocks torch DLLs.

Run on Google Colab (T4) or an unconstrained machine for full coverage.
Tests use randomly initialised models (no pretrained weight download).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from conftest import requires_torch  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_tiny_model(num_classes: int = 2):
    from src.detector import create_model
    return create_model(
        architecture="efficientnet_b3",
        pretrained=False,
        num_classes=num_classes,
        dropout_rate=0.0,
    )


def _make_random_input(batch: int = 1, size: int = 32):
    import torch
    return torch.randn(batch, 3, size, size)


def _save_fake_png(tmp_path: Path, width: int = 64, height: int = 64) -> Path:
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "test.png"
    img.save(p)
    return p


# ── Device tests ────────────────────────────────────────────────────────────

@requires_torch
class TestDevice:
    def test_resolve_auto_returns_device(self):
        import torch
        from src.detector import resolve_device
        d = resolve_device("auto")
        assert isinstance(d, torch.device)

    def test_resolve_cpu(self):
        import torch
        from src.detector import resolve_device
        d = resolve_device("cpu")
        assert d.type == "cpu"


# ── Model creation tests ────────────────────────────────────────────────────

@requires_torch
class TestCreateModel:
    def test_creates_model(self):
        model = _make_tiny_model()
        assert model is not None

    def test_model_is_nn_module(self):
        import torch.nn as nn
        model = _make_tiny_model()
        assert isinstance(model, nn.Module)

    def test_forward_pass_shape_2class(self):
        import torch
        model = _make_tiny_model(num_classes=2)
        model.eval()
        x = _make_random_input(batch=2, size=32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"

    def test_forward_pass_shape_1class(self):
        import torch
        model = _make_tiny_model(num_classes=1)
        model.eval()
        x = _make_random_input(batch=2, size=32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"

    def test_invalid_architecture_raises(self):
        from src.detector import create_model
        with pytest.raises(RuntimeError):
            create_model(architecture="not_a_real_model_xyz", pretrained=False)


# ── Checkpoint tests ────────────────────────────────────────────────────────

@requires_torch
class TestCheckpoint:
    def test_save_and_load_checkpoint(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, load_checkpoint
        model = _make_tiny_model()
        ckpt_path = tmp_path / "test_checkpoint.pt"

        save_checkpoint(
            path=ckpt_path, model=model, epoch=3,
            val_metrics={"roc_auc": 0.85, "accuracy": 0.80},
            train_cfg={"batch_size": 16, "learning_rate": 0.001},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        assert ckpt_path.exists()

        loaded_model, ckpt_meta = load_checkpoint(ckpt_path, device=torch.device("cpu"))
        assert ckpt_meta["epoch"] == 3
        assert abs(ckpt_meta["val_metrics"]["roc_auc"] - 0.85) < 1e-6

    def test_loaded_model_produces_same_outputs(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, load_checkpoint
        model = _make_tiny_model()
        model.eval()
        x = _make_random_input(batch=1, size=32)
        with torch.no_grad():
            original_out = model(x).clone()

        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={},
            train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        from src.detector import load_checkpoint
        loaded_model, _ = load_checkpoint(ckpt_path, device=torch.device("cpu"))
        loaded_model.eval()
        with torch.no_grad():
            loaded_out = loaded_model(x)
        assert torch.allclose(original_out, loaded_out, atol=1e-5)

    def test_load_missing_checkpoint_raises(self, tmp_path):
        from src.detector import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_checkpoint_contains_required_keys(self, tmp_path):
        import torch
        from src.detector import save_checkpoint
        model = _make_tiny_model()
        ckpt_path = tmp_path / "req_keys.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=1,
            val_metrics={"roc_auc": 0.7},
            train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=99,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "model_config", "epoch", "val_metrics", "seed"):
            assert key in ckpt, f"Missing checkpoint key: {key}"


# ── Single-image inference tests ────────────────────────────────────────────

@requires_torch
class TestPredictSingle:
    def test_predict_single_output_keys(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, predict_single
        model = _make_tiny_model()
        ckpt_path = tmp_path / "infer_ckpt.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={},
            train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2,
                       "dropout_rate": 0.0, "input_size": 32},
            seed=42,
        )
        image_path = _save_fake_png(tmp_path)
        result = predict_single(
            image_path=image_path, checkpoint_path=ckpt_path,
            device=torch.device("cpu"), image_size=32,
        )
        for key in ("predicted_class", "predicted_label", "synthetic_probability",
                    "real_probability", "logit", "note"):
            assert key in result

    def test_predict_returns_valid_probability(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, predict_single
        model = _make_tiny_model()
        ckpt_path = tmp_path / "prob_ckpt.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={}, train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        image_path = _save_fake_png(tmp_path)
        result = predict_single(image_path, ckpt_path, device=torch.device("cpu"), image_size=32)
        p = result["synthetic_probability"]
        assert 0.0 <= p <= 1.0

    def test_predict_label_is_real_or_synthetic(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, predict_single
        model = _make_tiny_model()
        ckpt_path = tmp_path / "label_ckpt.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={}, train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        image_path = _save_fake_png(tmp_path)
        result = predict_single(image_path, ckpt_path, device=torch.device("cpu"), image_size=32)
        assert result["predicted_label"] in ("real", "synthetic")

    def test_note_mentions_not_calibrated(self, tmp_path):
        import torch
        from src.detector import save_checkpoint, predict_single
        model = _make_tiny_model()
        ckpt_path = tmp_path / "note_ckpt.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={}, train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        image_path = _save_fake_png(tmp_path)
        result = predict_single(image_path, ckpt_path, device=torch.device("cpu"), image_size=32)
        note = result.get("note", "").lower()
        assert "not" in note and "calibrat" in note


# ── Detector class tests ─────────────────────────────────────────────────────

@requires_torch
class TestDetectorClass:
    def test_not_loaded_initially(self):
        from src.detector import Detector
        d = Detector()
        assert not d.is_loaded

    def test_load_marks_as_loaded(self, tmp_path):
        import torch
        from src.detector import Detector, save_checkpoint
        model = _make_tiny_model()
        ckpt_path = tmp_path / "det_ckpt.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0, val_metrics={}, train_cfg={},
            model_cfg={"architecture": "efficientnet_b3", "num_classes": 2, "dropout_rate": 0.0},
            seed=42,
        )
        d = Detector()
        d.load(ckpt_path, device=torch.device("cpu"), image_size=32)
        assert d.is_loaded

    def test_predict_raises_when_not_loaded(self, tmp_path):
        from src.detector import Detector
        from PIL import Image
        import numpy as np
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image_path = tmp_path / "img.png"
        Image.fromarray(arr).save(image_path)
        d = Detector()
        with pytest.raises(RuntimeError, match="not loaded"):
            d.predict(image_path)
