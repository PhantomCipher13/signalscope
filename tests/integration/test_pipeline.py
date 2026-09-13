"""
Phase 1 integration test — End-to-end mini-training pipeline.

All tests in this file require torch. They are automatically skipped
when a Windows Application Control policy blocks torch DLLs.
Run on Google Colab (T4) or an unconstrained machine for full coverage.

This is a software pipeline test — not an ML benchmark.
All metrics from this test are on random-noise images and have
no scientific meaning whatsoever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from conftest import requires_torch  # noqa: E402


# ── Fixture: tiny synthetic dataset ────────────────────────────────────────────

@pytest.fixture
def tiny_dataset(tmp_path):
    """20 real + 20 fake 32×32 PNG images — software fixture only."""
    from PIL import Image
    import numpy as np

    root = tmp_path / "dataset"
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    for i in range(20):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(real_dir / f"real_{i:04d}.png")
        arr2 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr2).save(fake_dir / f"fake_{i:04d}.png")

    return root


@requires_torch
class TestEndToEndPipeline:
    """Full pipeline on tiny synthetic data. Software test only — no ML benchmarks."""

    def test_full_pipeline(self, tiny_dataset, tmp_path):
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader

        from src.dataset import (
            create_manifest, load_manifest, filter_split, SignalScopeDataset
        )
        from src.detector import create_model, load_checkpoint, save_checkpoint
        from src.evaluation import evaluate_dataloader
        from src.preprocessing import build_val_transforms, build_train_transforms

        device = torch.device("cpu")
        image_size = 32

        # Step 1: manifest
        manifest_path = tmp_path / "manifest.csv"
        summary = create_manifest(
            root_dir=tiny_dataset, output_path=manifest_path,
            seed=42, train_frac=0.6, val_frac=0.2, test_frac=0.2,
            skip_invalid=False,
        )
        assert summary["total"] == 40

        # Step 2: datasets
        all_samples = load_manifest(manifest_path)
        train_samples = filter_split(all_samples, "train")
        val_samples = filter_split(all_samples, "val")
        assert len(train_samples) > 0
        assert len(val_samples) > 0

        train_ds = SignalScopeDataset(
            train_samples,
            transform=build_train_transforms(image_size=image_size, jpeg_prob=0.0, resize_prob=0.0)
        )
        val_ds = SignalScopeDataset(val_samples, transform=build_val_transforms(image_size=image_size))
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        # Step 3: model (random weights, no download)
        model = create_model(
            architecture="efficientnet_b3", pretrained=False,
            num_classes=2, dropout_rate=0.0,
        ).to(device)

        # Step 4: train 1 epoch
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Step 5: save checkpoint
        ckpt_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(
            path=ckpt_path, model=model, epoch=0,
            val_metrics={"roc_auc": 0.5},
            train_cfg={"batch_size": 4, "learning_rate": 1e-3},
            model_cfg={
                "architecture": "efficientnet_b3", "num_classes": 2,
                "dropout_rate": 0.0, "input_size": image_size,
            },
            seed=42,
        )
        assert ckpt_path.exists()

        # Step 6: load checkpoint
        loaded_model, ckpt_meta = load_checkpoint(ckpt_path, device=device)
        assert ckpt_meta["epoch"] == 0

        # Step 7: evaluate
        loaded_model.eval()
        result = evaluate_dataloader(loaded_model, val_loader, device, return_predictions=True)
        metrics = result["metrics"]
        for key in ("roc_auc", "accuracy", "f1"):
            assert key in metrics
        assert 0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] == len(val_samples)

        # Step 8: single-image predict
        from src.detector import predict_single
        first_val_path = Path(val_samples[0]["path"])
        pred = predict_single(
            image_path=first_val_path, checkpoint_path=ckpt_path,
            device=device, image_size=image_size,
        )
        assert 0.0 <= pred["synthetic_probability"] <= 1.0
        assert pred["predicted_label"] in ("real", "synthetic")
        assert "not" in pred["note"].lower()

        print(
            f"\n[Pipeline test: random noise — not meaningful] "
            f"accuracy={metrics['accuracy']:.3f}"
        )
