#!/usr/bin/env python3
"""
SignalScope — scripts/inspect_dataset.py
=========================================
Dataset inspection: discovers images, infers labels, and reports
statistics without modifying anything.

Usage:
    python scripts/inspect_dataset.py --root <dataset_dir>
    python scripts/inspect_dataset.py --root <dataset_dir> --check-images

Output:
    - Directory structure summary
    - File counts by extension
    - Inferred label distribution
    - Generator hints (from directory names)
    - Duplicate filename warnings
    - Sample dimension statistics (when --check-images is passed)
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import SUPPORTED_EXTENSIONS, load_image
from src.dataset import (
    LABEL_REAL, LABEL_SYNTHETIC,
    _DEFAULT_REAL_KEYWORDS, _DEFAULT_SYNTHETIC_KEYWORDS,
    discover_images, infer_generator, infer_label,
)
from src.utils import get_logger

logger = get_logger("inspect_dataset")


def inspect(root_dir: Path, check_images: bool = False) -> None:
    print(f"\n=== SignalScope Dataset Inspection ===")
    print(f"Root: {root_dir.resolve()}\n")

    if not root_dir.is_dir():
        print(f"[ERROR] Not a directory: {root_dir}")
        sys.exit(1)

    # ── Directory structure ────────────────────────────────────────────────
    print("--- Directory Structure (depth <= 3) ---")
    _print_tree(root_dir, max_depth=3)
    print()

    # ── Image discovery ────────────────────────────────────────────────────
    print("--- Discovering images... ---")
    images = discover_images(root_dir)
    if not images:
        print("[WARNING] No image files found. Check --root and supported extensions.")
        print(f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        return

    # Extension distribution
    ext_counter: dict[str, int] = collections.Counter(p.suffix.lower() for p in images)
    print(f"\nTotal images found: {len(images)}")
    print("By extension:")
    for ext, count in sorted(ext_counter.items()):
        print(f"  {ext:10s}  {count:>8d}")

    # ── Label inference ────────────────────────────────────────────────────
    print("\n--- Label Inference ---")
    real_count = 0
    synthetic_count = 0
    unknown_count = 0
    label_errors: list[str] = []
    generator_counter: dict[str, int] = collections.Counter()

    for img_path in images:
        try:
            label, class_dir = infer_label(img_path)
            if label == LABEL_REAL:
                real_count += 1
            else:
                synthetic_count += 1
            gen = infer_generator(img_path)
            if gen:
                generator_counter[gen] += 1
        except ValueError:
            unknown_count += 1
            if len(label_errors) < 5:
                label_errors.append(str(img_path))

    total_labelled = real_count + synthetic_count
    print(f"  Real:              {real_count:>8d}")
    print(f"  Synthetic:         {synthetic_count:>8d}")
    print(f"  Unknown / unlabelled: {unknown_count:>5d}")
    if unknown_count > 0:
        print(f"  (first few unknown paths):")
        for p in label_errors:
            print(f"    {p}")
        print(f"  -> Update real_dir_keywords / synthetic_dir_keywords in default.yaml")

    if real_count == 0 or synthetic_count == 0:
        print("\n[WARNING] One class has zero samples. Check directory naming.")

    # ── Generator hints ───────────────────────────────────────────────────
    if generator_counter:
        print("\n--- Generator Hints (from subdirectory names) ---")
        print("  NOTE: These are directory names only. Verify they are actual")
        print("  generator identifiers before using generator-separated splits.")
        for gen_name, count in sorted(generator_counter.items(), key=lambda x: -x[1]):
            print(f"  {gen_name:30s}  {count:>6d}")
    else:
        print("\n--- Generator Hints: none detected (flat directory structure) ---")
        print("  Generator-separated splits will not be available.")

    # ── Duplicate filename check ──────────────────────────────────────────
    print("\n--- Duplicate Filename Check ---")
    name_map: dict[str, list[Path]] = collections.defaultdict(list)
    for p in images:
        name_map[p.name].append(p)
    duplicates = {k: v for k, v in name_map.items() if len(v) > 1}
    if duplicates:
        print(f"  [WARNING] {len(duplicates)} duplicate filenames found:")
        for name, paths in list(duplicates.items())[:10]:
            print(f"    {name}:")
            for dp in paths:
                print(f"      {dp}")
    else:
        print(f"  No duplicate filenames detected.")

    # ── Image dimension statistics (optional) ─────────────────────────────
    if check_images:
        print("\n--- Image Dimension Statistics (sampling up to 500) ---")
        sample_paths = images[:500]
        widths, heights = [], []
        corrupt = 0
        for p in sample_paths:
            try:
                img = load_image(p)
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except OSError:
                corrupt += 1

        if corrupt:
            print(f"  [WARNING] {corrupt} corrupt/unloadable images in sample.")
        if widths:
            import numpy as np
            print(f"  Sampled {len(widths)} images:")
            print(f"  Width:   min={min(widths)} max={max(widths)} "
                  f"mean={np.mean(widths):.1f} median={int(np.median(widths))}")
            print(f"  Height:  min={min(heights)} max={max(heights)} "
                  f"mean={np.mean(heights):.1f} median={int(np.median(heights))}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n=== Inspection Complete ===")
    print(f"  Total images:      {len(images)}")
    print(f"  Labelled:          {total_labelled} "
          f"(real={real_count}, synthetic={synthetic_count})")
    if unknown_count > 0:
        print(f"  [ACTION REQUIRED] {unknown_count} images with unknown labels.")
        print(f"    Update default.yaml: dataset.real_dir_keywords and "
              f"synthetic_dir_keywords")
    print()


def _print_tree(root: Path, max_depth: int = 3, prefix: str = "") -> None:
    """Print a simplified directory tree."""
    try:
        children = sorted(root.iterdir())
    except PermissionError:
        print(f"{prefix}[Permission denied]")
        return

    dirs = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file()]

    depth = len(prefix) // 2
    if depth >= max_depth:
        if dirs:
            print(f"{prefix}  ... ({len(dirs)} subdirs, {len(files)} files)")
        return

    for d in dirs:
        file_count = sum(1 for f in d.rglob("*") if f.is_file()
                         and f.suffix.lower() in SUPPORTED_EXTENSIONS)
        print(f"{prefix}[{d.name}/]  ({file_count} images)")
        _print_tree(d, max_depth=max_depth, prefix=prefix + "  ")

    if files:
        img_files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if img_files:
            print(f"{prefix}  {len(img_files)} image file(s) here")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a dataset directory for SignalScope."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also load and check image dimensions (slower).",
    )
    args = parser.parse_args()
    inspect(args.root, check_images=args.check_images)


if __name__ == "__main__":
    main()
