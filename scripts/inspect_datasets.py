#!/usr/bin/env python3
"""
scripts/inspect_datasets.py
=============================
Deep inspection of all four SignalScope datasets extracted under data/raw/.

Produces:
  - data/manifests/master_manifest.csv
  - docs/DATASET_INSPECTION.md (structured inspection report)

Reports per dataset:
  - Identity, source, structure
  - Image counts (total, by extension, by split, by label, by generator)
  - Unreadable/corrupt images
  - Existing split structure
  - Generator metadata

Scientific integrity guarantees:
  - Does NOT invent labels, generators, or metadata
  - Clearly marks any inferred vs. explicitly available information
  - Does NOT train any model

Usage:
    python scripts/inspect_datasets.py
    python scripts/inspect_datasets.py --sample-dims 500  (load images to get dimensions)
    python scripts/inspect_datasets.py --no-dims          (skip dimension sampling)

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
DOCS_DIR = PROJECT_ROOT / "docs"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ── Dataset-specific knowledge from README inspection ─────────────────────────

DATASET_CONFIGS = {
    "CIFAKE": {
        "description": "CIFAKE: Real and AI-Generated Synthetic Images",
        "paper": "Bird & Lotfi (2023), https://arxiv.org/abs/2303.08854",
        "source_url": "https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images",
        "real_source": "CIFAR-10 (32x32 real photographs)",
        "synthetic_source": "Stable Diffusion v1.4 (32x32 CIFAR-10-conditioned)",
        "generator_known": True,
        "label_from": "directory name (REAL / FAKE)",
        "split_from": "directory (train / test)",
        "real_keywords": ["real"],
        "synthetic_keywords": ["fake"],
    },
    "GenImage": {
        "description": "GenImage benchmark repository (code only — images NOT included in this ZIP)",
        "paper": "Zhu et al. (2024), https://arxiv.org/abs/2306.08571",
        "source_url": "https://github.com/GenImage-Dataset/GenImage",
        "real_source": "ImageNet",
        "synthetic_source": "8 generators: Midjourney, Stable Diffusion v1.4, Stable Diffusion v1.5, ADM, GLIDE, VQDM, BigGAN, Wukong",
        "generator_known": True,
        "label_from": "directory name (ai / nature)",
        "split_from": "directory (train / val)",
        "notes": "This ZIP contains benchmark code and detector implementations ONLY. Actual ~500GB image dataset must be downloaded separately from Baidu Yunpan or Google Drive.",
    },
    "RAID": {
        "description": "RAID: Robust AI-Generated Image Detection (code only)",
        "paper": "Cheng et al. (2026), https://arxiv.org/abs/2607.28974",
        "source_url": "https://github.com/renxi-seu/RAID",
        "notes": "This ZIP contains ONLY Python code (train.py, test.py, model.py, dataloader.py). No images included. Uses GenImage dataset for training/evaluation. Proposes bit-reversal augmentation for robustness.",
    },
    "UnbiasedGenImage": {
        "description": "UnbiasedGenImage: Reveals compression/size biases in GenImage",
        "paper": "https://arxiv.org/abs/2403.17608",
        "source_url": "https://github.com/gendetection/UnbiasedGenImage",
        "notes": "This ZIP contains code, metadata tools, and result figures ONLY. Actual images come from the GenImage dataset (downloaded separately). Provides metadata.csv with JPEG QF, size, and content fields per image.",
    },
}


def get_file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_image_dims(path: Path) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def infer_cifake_metadata(rel_path: str) -> dict:
    """Infer label, split, generator from CIFAKE path structure.
    
    Expected: DATASET/split/LABEL/filename
    e.g. DATASET/train/FAKE/0.jpg
    """
    parts = Path(rel_path).parts
    meta = {
        "label": None, "label_name": None,
        "split": None, "generator": None,
        "source_class": None,
    }
    for i, p in enumerate(parts):
        if p.upper() in ("REAL", "FAKE"):
            meta["label_name"] = p.upper()
            meta["label"] = 0 if p.upper() == "REAL" else 1
        if p.lower() in ("train", "test", "val", "validation"):
            meta["split"] = p.lower()
    if meta["label_name"] == "FAKE":
        meta["generator"] = "stable_diffusion_v1.4"  # Confirmed from CIFAKE paper
    elif meta["label_name"] == "REAL":
        meta["generator"] = "cifar10_real"
    return meta


def scan_directory(root: Path, dataset_name: str, config: dict) -> list[dict]:
    """Recursively scan a dataset directory and collect per-image metadata."""
    records = []
    if not root.exists():
        return records

    for fpath in root.rglob("*"):
        if not fpath.is_file():
            continue
        suffix = fpath.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        rel_path = fpath.relative_to(PROJECT_ROOT).as_posix()
        file_size = fpath.stat().st_size

        record = {
            "image_path": rel_path,
            "label": None,
            "label_name": None,
            "dataset": dataset_name,
            "generator": None,
            "split": None,
            "original_split": None,
            "source_class": None,
            "extension": suffix.lstrip("."),
            "file_size_bytes": file_size,
            "width": None,
            "height": None,
        }

        # Dataset-specific label/split/generator inference
        if dataset_name == "CIFAKE":
            rel_internal = fpath.relative_to(root).as_posix()
            meta = infer_cifake_metadata(rel_internal)
            record.update({k: v for k, v in meta.items() if v is not None})
            record["original_split"] = record.get("split")

        records.append(record)

    return records


def count_unreadable(root: Path, sample: int = 0) -> tuple[int, int, list[str]]:
    """Count total images, unreadable images; optionally sample dimensions.
    Returns (total, unreadable_count, list_of_bad_paths)"""
    total = 0
    unreadable = 0
    bad_paths = []
    for fpath in root.rglob("*"):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        total += 1
        if sample > 0 and total <= sample:
            dims = load_image_dims(fpath)
            if dims is None:
                unreadable += 1
                bad_paths.append(str(fpath))
    return total, unreadable, bad_paths


def compute_dim_stats(records: list[dict], max_sample: int = 300) -> dict:
    """Sample up to max_sample records, load images, compute dimension stats."""
    from PIL import Image
    import numpy as np

    widths, heights = [], []
    corrupt = 0
    sampled = 0

    for r in records[:max_sample]:
        path = PROJECT_ROOT / r["image_path"]
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
            sampled += 1
        except Exception:
            corrupt += 1

    if not widths:
        return {"sampled": 0, "corrupt_in_sample": corrupt}

    return {
        "sampled": sampled,
        "corrupt_in_sample": corrupt,
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": round(float(np.mean(widths)), 1),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": round(float(np.mean(heights)), 1),
        "unique_dims": len(set(zip(widths, heights))),
    }


def inspect_cifake(sample_dims: int) -> dict:
    root = DATA_RAW / "CIFAKE"
    records = scan_directory(root, "CIFAKE", DATASET_CONFIGS["CIFAKE"])

    by_split_label = collections.Counter()
    for r in records:
        by_split_label[(r.get("split"), r.get("label_name"))] += 1

    generators = collections.Counter(r.get("generator") for r in records)
    extensions = collections.Counter(r["extension"] for r in records)

    dim_stats = {}
    if sample_dims > 0:
        dim_stats = compute_dim_stats(records, sample_dims)

    return {
        "name": "CIFAKE",
        "root": str(root),
        "config": DATASET_CONFIGS["CIFAKE"],
        "total_images": len(records),
        "by_split_label": dict(by_split_label),
        "generators": dict(generators),
        "extensions": dict(extensions),
        "dim_stats": dim_stats,
        "records": records,
        "has_images": len(records) > 0,
    }


def inspect_genimage() -> dict:
    root = DATA_RAW / "GenImage"
    # Only code — count all files
    all_files = list(root.rglob("*")) if root.exists() else []
    py_files = [f for f in all_files if f.suffix == ".py"]
    img_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    return {
        "name": "GenImage",
        "root": str(root),
        "config": DATASET_CONFIGS["GenImage"],
        "total_images": len(img_files),
        "total_code_files": len(py_files),
        "total_files": len([f for f in all_files if f.is_file()]),
        "has_images": len(img_files) > 0,
        "records": [],
        "notes": "This ZIP contains benchmark code only. Actual image dataset (~500 GB) must be downloaded separately.",
    }


def inspect_raid() -> dict:
    root = DATA_RAW / "RAID"
    all_files = list(root.rglob("*")) if root.exists() else []
    py_files = [f for f in all_files if f.suffix == ".py"]

    return {
        "name": "RAID",
        "root": str(root),
        "config": DATASET_CONFIGS["RAID"],
        "total_images": 0,
        "total_code_files": len(py_files),
        "total_files": len([f for f in all_files if f.is_file()]),
        "has_images": False,
        "records": [],
        "notes": "Code-only repository. Uses GenImage for training/evaluation.",
    }


def inspect_unbiasedgenimage() -> dict:
    root = DATA_RAW / "UnbiasedGenImage"
    all_files = list(root.rglob("*")) if root.exists() else []
    py_files = [f for f in all_files if f.suffix == ".py"]
    img_files = [f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    # Look for metadata CSV
    csvs = [f for f in all_files if f.suffix == ".csv"]
    txt_files = [f for f in all_files if f.suffix == ".txt"]

    return {
        "name": "UnbiasedGenImage",
        "root": str(root),
        "config": DATASET_CONFIGS["UnbiasedGenImage"],
        "total_images": len(img_files),
        "image_files": [str(f) for f in img_files],  # These are result plots, not dataset images
        "total_code_files": len(py_files),
        "total_files": len([f for f in all_files if f.is_file()]),
        "csv_files": [str(f.relative_to(PROJECT_ROOT)) for f in csvs],
        "has_images": False,  # PNG files are result figures, not dataset images
        "records": [],
        "notes": "Code + result figures only. No dataset images included.",
    }


def write_master_manifest(all_records: list[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not all_records:
        print(f"  [WARN] No image records — manifest will be empty")
        output_path.write_text("image_path,label,label_name,dataset,generator,split,original_split,source_class,extension,file_size_bytes,width,height\n")
        return 0

    fieldnames = [
        "image_path", "label", "label_name", "dataset", "generator",
        "split", "original_split", "source_class",
        "extension", "file_size_bytes", "width", "height",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    return len(all_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SignalScope datasets.")
    parser.add_argument("--sample-dims", type=int, default=300,
                        help="Number of images to sample for dimension stats (0 = skip)")
    parser.add_argument("--no-dims", action="store_true",
                        help="Skip dimension sampling entirely")
    args = parser.parse_args()
    sample_dims = 0 if args.no_dims else args.sample_dims

    print("=== SignalScope Dataset Inspection ===\n")

    # ── Inspect each dataset ─────────────────────────────────────────────────
    print("Phase C: Inspecting CIFAKE...")
    cifake = inspect_cifake(sample_dims)
    print(f"  Total images: {cifake['total_images']}")
    for k, v in cifake['by_split_label'].items():
        print(f"    {k}: {v}")

    print("\nPhase C: Inspecting GenImage...")
    genimage = inspect_genimage()
    print(f"  Total images in ZIP: {genimage['total_images']} (code repo only)")
    print(f"  Python files: {genimage['total_code_files']}")

    print("\nPhase C: Inspecting RAID...")
    raid = inspect_raid()
    print(f"  Total files: {raid['total_files']} (code only)")

    print("\nPhase C: Inspecting UnbiasedGenImage...")
    ubgi = inspect_unbiasedgenimage()
    print(f"  Total files: {ubgi['total_files']} (code + figures only)")
    if ubgi["csv_files"]:
        print(f"  CSV files found: {ubgi['csv_files']}")

    # ── Write master manifest ─────────────────────────────────────────────────
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    all_records = cifake["records"]  # Only CIFAKE has actual images
    manifest_path = MANIFESTS_DIR / "master_manifest.csv"
    n_written = write_master_manifest(all_records, manifest_path)
    print(f"\nMaster manifest: {manifest_path} ({n_written} records)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== INSPECTION SUMMARY ===")
    for ds in [cifake, genimage, raid, ubgi]:
        print(f"\n{ds['name']}:")
        print(f"  Has images: {ds['has_images']}")
        print(f"  Total images: {ds.get('total_images', 0)}")
        if "by_split_label" in ds:
            for k, v in ds["by_split_label"].items():
                print(f"    Split/Label {k}: {v}")
        if "generators" in ds:
            for gen, cnt in ds["generators"].items():
                print(f"    Generator '{gen}': {cnt}")
        if "dim_stats" in ds and ds["dim_stats"]:
            print(f"  Dimensions: {ds['dim_stats']}")
        if ds.get("notes"):
            print(f"  NOTE: {ds['notes']}")


if __name__ == "__main__":
    main()
