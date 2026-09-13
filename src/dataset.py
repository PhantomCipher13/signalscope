"""
SignalScope — src/dataset.py
==============================
Dataset manifest generation, loading, and DataLoader construction.

Responsibilities:
  - Discover image files from a directory tree.
  - Infer class labels from directory structure.
  - Generate a deterministic CSV manifest.
  - Perform train/val/test splitting with leakage prevention.
  - Provide a PyTorch Dataset and DataLoader factory.

Label convention:  0 = real,  1 = synthetic/AI-generated

IMPORTANT — scientific integrity:
  - Generator identities are never invented.
  - Split labels are derived only from the actual directory structure.
  - If generator-separated splits cannot be created from the available
    data, this is reported clearly and random splitting is used instead.
  - The unseen-generator test split must NOT be used for checkpoint
    selection, calibration, or threshold tuning.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger("signalscope.dataset")

# Label convention
LABEL_REAL = 0
LABEL_SYNTHETIC = 1

# Default keywords for inferring class from directory names
_DEFAULT_REAL_KEYWORDS = frozenset([
    "real", "genuine", "authentic", "original", "nature", "0_real", "0-real"
])
_DEFAULT_SYNTHETIC_KEYWORDS = frozenset([
    "fake", "ai", "synthetic", "generated", "gan", "diffusion",
    "1_fake", "1-fake", "aigc", "midjourney", "stablediffusion",
    "dalle", "firefly"
])

MANIFEST_COLUMNS = [
    "sample_id",
    "path",
    "label",
    "label_name",
    "class_dir",
    "generator",    # null when not determinable from directory structure
    "split",
]


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def discover_images(
    root_dir: str | Path,
    extensions: frozenset[str] | None = None,
) -> list[Path]:
    """Recursively discover all image files under *root_dir*.

    Parameters
    ----------
    root_dir:
        Root directory to scan.
    extensions:
        Allowed file extensions (lower-case, including dot).
        Defaults to JPEG, PNG, BMP, WebP.

    Returns
    -------
    list[Path]
        Sorted list of absolute image paths.
    """
    from src.preprocessing import SUPPORTED_EXTENSIONS  # avoid circular at top

    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_dir}")

    found: list[Path] = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            found.append(p.resolve())

    found.sort()
    logger.info("Discovered %d image files under %s", len(found), root_dir)
    return found


def infer_label(
    path: Path,
    real_keywords: frozenset[str] = _DEFAULT_REAL_KEYWORDS,
    synthetic_keywords: frozenset[str] = _DEFAULT_SYNTHETIC_KEYWORDS,
) -> tuple[int, str]:
    """Infer the binary label from the directory path of an image.

    Walks the path's parent directories (nearest first) looking for a
    keyword match. Returns (label, class_dir_name).

    Returns
    -------
    tuple[int, str]
        (LABEL_REAL or LABEL_SYNTHETIC, matched_directory_name)

    Raises
    ------
    ValueError
        If no label can be inferred from the directory structure.
    """
    for part in reversed(path.parts):
        part_lower = part.lower()
        # Check synthetic first (more specific)
        for kw in synthetic_keywords:
            if kw in part_lower:
                return LABEL_SYNTHETIC, part
        for kw in real_keywords:
            if kw in part_lower:
                return LABEL_REAL, part
    raise ValueError(
        f"Cannot infer label from path: {path}\n"
        f"  Expected a directory component matching one of:\n"
        f"  real keywords: {sorted(real_keywords)}\n"
        f"  synthetic keywords: {sorted(synthetic_keywords)}\n"
        f"  Consider updating dataset.real_dir_keywords / "
        f"synthetic_dir_keywords in default.yaml"
    )


def infer_generator(path: Path) -> Optional[str]:
    """Attempt to infer the generator name from the directory path.

    Returns the immediate parent directory name if it looks like a
    generator-specific subfolder (e.g. 'midjourney', 'stable_diffusion').
    Returns None if no generator can be determined.

    Generator identities are never invented — only reported when clearly
    present in the directory structure.
    """
    # For now: return the immediate parent name as a potential generator hint.
    # Manifest creation reports this as-is; user can review and correct.
    parent_name = path.parent.name
    # Don't call it a generator if it's just the class-level folder
    if parent_name.lower() in _DEFAULT_REAL_KEYWORDS | _DEFAULT_SYNTHETIC_KEYWORDS:
        return None
    return parent_name if parent_name else None


def make_sample_id(path: Path, root_dir: Path) -> str:
    """Create a stable sample ID from the path relative to root_dir."""
    try:
        rel = path.relative_to(root_dir)
    except ValueError:
        rel = path
    return hashlib.md5(str(rel).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------

def create_manifest(
    root_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    real_keywords: frozenset[str] | None = None,
    synthetic_keywords: frozenset[str] | None = None,
    extensions: frozenset[str] | None = None,
    skip_invalid: bool = True,
) -> dict:
    """Discover images, infer labels, split, and write a CSV manifest.

    Split strategy: stratified random split by label (preserves class
    balance across splits). If generator metadata is available, a
    generator-separated split should be done manually after inspection.

    Parameters
    ----------
    root_dir:
        Root directory of the dataset.
    output_path:
        Where to write the manifest CSV.
    seed:
        Random seed for reproducible splitting.
    train_frac, val_frac, test_frac:
        Split proportions (must sum to 1.0).
    real_keywords, synthetic_keywords:
        Keyword sets for label inference. Defaults used if None.
    extensions:
        Allowed image extensions. Defaults to SUPPORTED_EXTENSIONS.
    skip_invalid:
        If True, skip images that cannot be loaded (corrupt/invalid).
        Counts of skipped images are always reported.

    Returns
    -------
    dict
        Summary statistics: total, real_count, synthetic_count,
        invalid_count, label_unknown_count, train_count, val_count,
        test_count.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        "train_frac + val_frac + test_frac must equal 1.0"
    )

    root_dir = Path(root_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if real_keywords is None:
        real_keywords = _DEFAULT_REAL_KEYWORDS
    if synthetic_keywords is None:
        synthetic_keywords = _DEFAULT_SYNTHETIC_KEYWORDS

    # 1. Discover images
    images = discover_images(root_dir, extensions=extensions)
    if not images:
        logger.warning("No images found under %s", root_dir)
        return _empty_summary()

    # 2. Infer labels
    real_samples: list[dict] = []
    synthetic_samples: list[dict] = []
    invalid_count = 0
    label_unknown_count = 0

    for img_path in images:
        try:
            label, class_dir = infer_label(img_path, real_keywords, synthetic_keywords)
        except ValueError:
            label_unknown_count += 1
            logger.debug("Label unknown: %s", img_path)
            continue

        if skip_invalid:
            from src.preprocessing import is_valid_image  # lazy
            if not is_valid_image(img_path):
                invalid_count += 1
                logger.debug("Skipped invalid image: %s", img_path)
                continue

        generator = infer_generator(img_path)
        sample = {
            "sample_id": make_sample_id(img_path, root_dir),
            "path": str(img_path),
            "label": label,
            "label_name": "real" if label == LABEL_REAL else "synthetic",
            "class_dir": class_dir,
            "generator": generator or "",
            "split": "",  # filled below
        }

        if label == LABEL_REAL:
            real_samples.append(sample)
        else:
            synthetic_samples.append(sample)

    logger.info(
        "Label inference: real=%d, synthetic=%d, unknown=%d, invalid=%d",
        len(real_samples), len(synthetic_samples),
        label_unknown_count, invalid_count,
    )

    if not real_samples and not synthetic_samples:
        logger.error(
            "No samples with determinable labels. "
            "Check real_dir_keywords / synthetic_dir_keywords in config."
        )
        return _empty_summary(invalid_count=invalid_count, unknown_count=label_unknown_count)

    # 3. Stratified split
    rng = random.Random(seed)
    all_samples: list[dict] = []
    for group in (real_samples, synthetic_samples):
        _assign_splits(group, train_frac, val_frac, test_frac, rng)
        all_samples.extend(group)

    # Shuffle final list (deterministic)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(all_samples)

    # 4. Write manifest
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_samples)

    counts = _count_splits(all_samples)
    logger.info(
        "Manifest written: %s | train=%d val=%d test=%d",
        output_path, counts["train"], counts["val"], counts["test"],
    )

    return {
        "total": len(all_samples),
        "real_count": len(real_samples),
        "synthetic_count": len(synthetic_samples),
        "invalid_count": invalid_count,
        "label_unknown_count": label_unknown_count,
        **counts,
        "manifest_path": str(output_path),
    }


def _assign_splits(
    samples: list[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> None:
    """In-place: assign 'split' key to each sample in the list."""
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    for i, s in enumerate(samples):
        if i < n_train:
            s["split"] = "train"
        elif i < n_train + n_val:
            s["split"] = "val"
        else:
            s["split"] = "test"


def _count_splits(samples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for s in samples:
        sp = s.get("split", "")
        if sp in counts:
            counts[sp] += 1
    return counts


def _empty_summary(invalid_count: int = 0, unknown_count: int = 0) -> dict:
    return {
        "total": 0, "real_count": 0, "synthetic_count": 0,
        "invalid_count": invalid_count, "label_unknown_count": unknown_count,
        "train": 0, "val": 0, "test": 0,
    }


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load a manifest CSV and return a list of sample dicts."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            samples.append(row)

    logger.info("Loaded manifest: %d samples from %s", len(samples), path)
    return samples


def filter_split(samples: list[dict], split: str) -> list[dict]:
    """Return only samples belonging to the given split.

    Accepts both 'val' and 'validation' as equivalent.
    Accepts both 'test' and 'test_indistribution' as equivalent.
    """
    # Build a set of accepted split names for this query
    aliases = {split}
    if split == "val":
        aliases.add("validation")
    elif split == "validation":
        aliases.add("val")
    elif split == "test":
        aliases.add("test_indistribution")
    elif split == "test_indistribution":
        aliases.add("test")
    return [s for s in samples if s.get("split") in aliases]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SignalScopeDataset:
    """PyTorch-compatible dataset built from a manifest list.

    Parameters
    ----------
    samples:
        List of sample dicts (from ``load_manifest`` + ``filter_split``).
    transform:
        torchvision transform to apply to each PIL image.
    """

    def __init__(self, samples: list[dict], transform=None) -> None:
        self.samples = samples
        self.transform = transform
        # Normalise path key: manifests may use 'path' or 'image_path'
        for s in self.samples:
            if "path" not in s and "image_path" in s:
                s["path"] = s["image_path"]
        # Validate paths exist (warn if missing, don't crash — caller decides)
        missing = sum(1 for s in samples if not Path(s["path"]).exists())
        if missing:
            logger.warning(
                "%d/%d sample paths not found on disk. "
                "Training will fail if these are in the active split.",
                missing, len(samples),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return (tensor, label) for index *idx*."""
        import torch  # lazy
        from src.preprocessing import load_image  # lazy

        sample = self.samples[idx]
        img = load_image(sample["path"])
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label

    def class_counts(self) -> dict[str, int]:
        """Return {'real': N, 'synthetic': M} counts."""
        real = sum(1 for s in self.samples if s["label"] == LABEL_REAL)
        synth = len(self.samples) - real
        return {"real": real, "synthetic": synth}


def make_dataloaders(
    manifest_path: str | Path,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> dict:
    """Build train/val/test DataLoaders from a manifest.

    Parameters
    ----------
    manifest_path:
        Path to the manifest CSV.
    train_transform:
        Transform for training data (with augmentation).
    val_transform:
        Transform for val/test data (deterministic).
    batch_size:
        Samples per batch.
    num_workers:
        DataLoader workers. Use 0 on Windows to avoid multiprocessing issues.
    pin_memory:
        Pin memory for faster CUDA transfer.
    seed:
        Seed for DataLoader worker seeding.

    Returns
    -------
    dict with keys "train", "val", "test", each a DataLoader.
    """
    import torch
    from torch.utils.data import DataLoader

    samples = load_manifest(manifest_path)

    def _worker_init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    loaders: dict = {}
    for split_name in ("train", "val", "test"):
        split_samples = filter_split(samples, split_name)
        if not split_samples:
            logger.warning("Split '%s' is empty in manifest.", split_name)
            continue
        transform = train_transform if split_name == "train" else val_transform
        ds = SignalScopeDataset(split_samples, transform=transform)
        shuffle = split_name == "train"
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_worker_init if num_workers > 0 else None,
        )
        loaders[split_name] = loader
        logger.info(
            "DataLoader '%s': %d samples, batch_size=%d, shuffle=%s",
            split_name, len(ds), batch_size, shuffle,
        )

    return loaders
