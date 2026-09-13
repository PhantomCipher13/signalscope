"""
Phase 1 unit tests — Dataset module.

Tests manifest creation, label inference, split logic, and DataLoader
construction using a tiny synthetic fixture directory.

The synthetic fixture images are for software validation only.
They must never be presented as ML benchmark results.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    LABEL_REAL, LABEL_SYNTHETIC,
    create_manifest, discover_images, filter_split,
    infer_label, load_manifest, make_sample_id,
    SignalScopeDataset,
)


# ── Fixture: tiny synthetic dataset ──────────────────────────────────────────

def _make_fixture_dataset(tmp_path: Path, n_real: int = 10, n_fake: int = 10) -> Path:
    """Create a minimal fixture dataset with real/fake subdirectories."""
    from PIL import Image
    import numpy as np

    root = tmp_path / "dataset"
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    for i in range(n_real):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(real_dir / f"real_{i:04d}.png")

    for i in range(n_fake):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(fake_dir / f"fake_{i:04d}.png")

    return root


# ── discover_images tests ─────────────────────────────────────────────────────

class TestDiscoverImages:
    def test_finds_all_images(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=5, n_fake=5)
        images = discover_images(root)
        assert len(images) == 10

    def test_returns_absolute_paths(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=3, n_fake=3)
        images = discover_images(root)
        assert all(p.is_absolute() for p in images)

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_images(empty) == []

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            discover_images(tmp_path / "no_such_dir")


# ── infer_label tests ──────────────────────────────────────────────────────────

class TestInferLabel:
    def test_infers_real_from_dir_name(self, tmp_path):
        p = tmp_path / "real" / "img.png"
        p.parent.mkdir()
        p.touch()
        label, class_dir = infer_label(p)
        assert label == LABEL_REAL

    def test_infers_synthetic_from_fake_dir(self, tmp_path):
        p = tmp_path / "fake" / "img.png"
        p.parent.mkdir()
        p.touch()
        label, class_dir = infer_label(p)
        assert label == LABEL_SYNTHETIC

    def test_infers_synthetic_from_ai_dir(self, tmp_path):
        p = tmp_path / "ai_generated" / "img.png"
        p.parent.mkdir()
        p.touch()
        label, class_dir = infer_label(p)
        assert label == LABEL_SYNTHETIC

    def test_unknown_dir_raises(self, tmp_path):
        # Use directory names that contain no real/synthetic keywords anywhere.
        # Also override the keyword sets so only specific words match.
        from src.dataset import _DEFAULT_REAL_KEYWORDS, _DEFAULT_SYNTHETIC_KEYWORDS
        p = tmp_path / "zzzunknownzzz" / "img.png"
        p.parent.mkdir(parents=True)
        p.touch()
        # Use restricted keyword sets that definitely won't match any part of tmp_path
        with pytest.raises(ValueError):
            infer_label(
                p,
                real_keywords=frozenset(["real_only_match"]),
                synthetic_keywords=frozenset(["fake_only_match"]),
            )


# ── create_manifest tests ──────────────────────────────────────────────────────

class TestCreateManifest:
    def test_manifest_created(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=10, n_fake=10)
        out = tmp_path / "manifest.csv"
        summary = create_manifest(
            root_dir=root, output_path=out,
            seed=42, train_frac=0.6, val_frac=0.2, test_frac=0.2,
            skip_invalid=False,
        )
        assert out.exists()
        assert summary["total"] == 20

    def test_manifest_has_correct_columns(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=5, n_fake=5)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, train_frac=0.6, val_frac=0.2, test_frac=0.2,
                        skip_invalid=False)
        with out.open() as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in ("sample_id", "path", "label", "label_name", "split"):
            assert col in cols, f"Missing column: {col}"

    def test_split_fractions_approximately_correct(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=50, n_fake=50)
        out = tmp_path / "manifest.csv"
        summary = create_manifest(root, out, seed=42,
                                  train_frac=0.7, val_frac=0.15, test_frac=0.15,
                                  skip_invalid=False)
        total = summary["total"]
        train_frac = summary["train"] / total
        val_frac = summary["val"] / total
        # Allow ±0.1 tolerance for small datasets
        assert abs(train_frac - 0.70) < 0.1
        assert abs(val_frac - 0.15) < 0.1

    def test_manifest_is_deterministic(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=20, n_fake=20)
        out1 = tmp_path / "m1.csv"
        out2 = tmp_path / "m2.csv"
        create_manifest(root, out1, seed=42, skip_invalid=False)
        create_manifest(root, out2, seed=42, skip_invalid=False)
        assert out1.read_text() == out2.read_text(), (
            "Manifest must be deterministic for the same seed"
        )

    def test_different_seeds_differ(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=20, n_fake=20)
        out1 = tmp_path / "seed1.csv"
        out2 = tmp_path / "seed2.csv"
        create_manifest(root, out1, seed=1, skip_invalid=False)
        create_manifest(root, out2, seed=2, skip_invalid=False)
        # Different seeds → different splits (with high probability for 40 samples)
        assert out1.read_text() != out2.read_text()

    def test_labels_correct_in_manifest(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=5, n_fake=5)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        for s in samples:
            if s["label_name"] == "real":
                assert s["label"] == LABEL_REAL
            else:
                assert s["label"] == LABEL_SYNTHETIC


# ── filter_split tests ─────────────────────────────────────────────────────────

class TestFilterSplit:
    def test_filter_returns_correct_split(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=20, n_fake=20)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        train = filter_split(samples, "train")
        val = filter_split(samples, "val")
        test = filter_split(samples, "test")
        assert len(train) + len(val) + len(test) == 40
        assert all(s["split"] == "train" for s in train)
        assert all(s["split"] == "val" for s in val)

    def test_no_sample_in_multiple_splits(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=20, n_fake=20)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        train_ids = {s["sample_id"] for s in filter_split(samples, "train")}
        val_ids = {s["sample_id"] for s in filter_split(samples, "val")}
        test_ids = {s["sample_id"] for s in filter_split(samples, "test")}
        assert not (train_ids & val_ids), "Leakage: same sample in train and val"
        assert not (train_ids & test_ids), "Leakage: same sample in train and test"
        assert not (val_ids & test_ids), "Leakage: same sample in val and test"


# ── SignalScopeDataset tests ───────────────────────────────────────────────────

class TestSignalScopeDataset:
    def test_dataset_len(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=5, n_fake=5)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        from src.preprocessing import build_val_transforms
        ds = SignalScopeDataset(samples, transform=build_val_transforms(image_size=32))
        assert len(ds) == 10

    def test_dataset_getitem_returns_tensor_label(self, tmp_path):
        import torch
        root = _make_fixture_dataset(tmp_path, n_real=5, n_fake=5)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        from src.preprocessing import build_val_transforms
        ds = SignalScopeDataset(samples, transform=build_val_transforms(image_size=32))
        img_tensor, label = ds[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 32, 32)
        assert label.item() in (0, 1)

    def test_class_counts(self, tmp_path):
        root = _make_fixture_dataset(tmp_path, n_real=6, n_fake=4)
        out = tmp_path / "manifest.csv"
        create_manifest(root, out, seed=42, skip_invalid=False)
        samples = load_manifest(out)
        ds = SignalScopeDataset(samples)
        counts = ds.class_counts()
        assert counts["real"] + counts["synthetic"] == 10
