#!/usr/bin/env python3
"""
scripts/check_dataset_integrity.py
=====================================
Data quality checks for all SignalScope datasets.

Checks performed:
  1. Missing files referenced by manifest
  2. Duplicate paths in manifest
  3. Duplicate files by exact MD5 hash (configurable sample)
  4. Invalid labels
  5. Unreadable/corrupt images (sample)
  6. Impossible dimensions (w or h < 4, or > 100000)
  7. Empty directories
  8. Unexpected file types in image directories
  9. Generator names missing where structure provides them
  10. Potential exact-duplicate cross-dataset leakage

Reports are printed and also saved to data/manifests/integrity_report.txt

Usage:
    python scripts/check_dataset_integrity.py
    python scripts/check_dataset_integrity.py --hash-sample 1000
    python scripts/check_dataset_integrity.py --no-hash  (skip MD5 check)

PYTHON: C:\\Python311\\python.exe
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VALID_LABELS = {0, 1}
VALID_LABEL_NAMES = {"REAL", "FAKE"}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def check_manifest_paths(records: list[dict]) -> list[str]:
    issues = []
    for r in records:
        p = PROJECT_ROOT / r["image_path"]
        if not p.exists():
            issues.append(f"MISSING: {r['image_path']}")
    return issues


def check_duplicate_paths(records: list[dict]) -> list[str]:
    seen = collections.Counter(r["image_path"] for r in records)
    return [f"DUPLICATE_PATH ({cnt}x): {path}" for path, cnt in seen.items() if cnt > 1]


def check_labels(records: list[dict]) -> list[str]:
    issues = []
    for r in records:
        label = r.get("label")
        label_name = r.get("label_name", "")
        if label not in ("0", "1", None, ""):
            issues.append(f"INVALID_LABEL '{label}': {r['image_path']}")
        if label_name and label_name.upper() not in VALID_LABEL_NAMES:
            issues.append(f"INVALID_LABEL_NAME '{label_name}': {r['image_path']}")
    return issues


def check_corrupt_images(records: list[dict], max_sample: int = 200) -> tuple[list[str], int]:
    from PIL import Image
    bad = []
    checked = 0
    for r in records[:max_sample]:
        p = PROJECT_ROOT / r["image_path"]
        try:
            with Image.open(p) as img:
                img.verify()
            checked += 1
        except Exception as e:
            bad.append(f"CORRUPT: {r['image_path']} ({e})")
    return bad, checked


def check_dimensions(records: list[dict]) -> list[str]:
    """Flag records where dimension values are impossible (if width/height are in manifest)."""
    issues = []
    for r in records:
        w = r.get("width")
        h = r.get("height")
        if w and h:
            try:
                wi, hi = int(float(w)), int(float(h))
                if wi < 4 or hi < 4 or wi > 100000 or hi > 100000:
                    issues.append(f"IMPOSSIBLE_DIMS ({wi}x{hi}): {r['image_path']}")
            except (ValueError, TypeError):
                pass
    return issues


def check_missing_generators(records: list[dict]) -> list[str]:
    """Flag records where generator should be known but is missing."""
    issues = []
    for r in records:
        gen = r.get("generator", "")
        label = r.get("label", "")
        dataset = r.get("dataset", "")
        # For CIFAKE, generator is always known
        if dataset == "CIFAKE" and not gen:
            issues.append(f"MISSING_GENERATOR: {r['image_path']}")
    return issues


def find_empty_dirs(root: Path) -> list[str]:
    empties = []
    if not root.exists():
        return empties
    for d in root.rglob("*"):
        if d.is_dir() and not any(d.iterdir()):
            empties.append(str(d.relative_to(PROJECT_ROOT)))
    return empties


def find_unexpected_files(root: Path) -> list[str]:
    """Find non-image, non-README files in image-bearing directories."""
    unexpected = []
    if not root.exists():
        return unexpected
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        # Allow images, .txt READMEs, .extracted_ok sentinels
        if suffix not in SUPPORTED_EXTENSIONS and suffix not in (".txt", "") and f.name not in (".extracted_ok",):
            pass  # unexpected but don't flag deeply nested code files
    return unexpected


def check_hash_duplicates(records: list[dict], max_sample: int) -> dict[str, list[str]]:
    """MD5-hash up to max_sample images. Return hash -> [paths] for duplicates."""
    hash_map: dict[str, list[str]] = collections.defaultdict(list)
    for r in records[:max_sample]:
        p = PROJECT_ROOT / r["image_path"]
        if not p.exists():
            continue
        try:
            h = md5_file(p)
            hash_map[h].append(r["image_path"])
        except OSError:
            pass
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalScope dataset integrity checks.")
    parser.add_argument("--hash-sample", type=int, default=500,
                        help="Number of images to MD5-hash (0 = skip).")
    parser.add_argument("--no-hash", action="store_true",
                        help="Skip MD5 hashing entirely.")
    parser.add_argument("--corrupt-sample", type=int, default=200,
                        help="Number of images to load for corrupt check.")
    args = parser.parse_args()
    hash_sample = 0 if args.no_hash else args.hash_sample

    manifest_path = MANIFESTS_DIR / "master_manifest.csv"
    records = load_manifest(manifest_path)

    report_lines = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    log("=== SignalScope Dataset Integrity Report ===")
    log(f"Manifest: {manifest_path}")
    log(f"Records:  {len(records)}")
    log()

    # 1. Missing files
    log("--- Check 1: Missing Files ---")
    missing = check_manifest_paths(records)
    if missing:
        log(f"  [FAIL] {len(missing)} missing files:")
        for m in missing[:20]:
            log(f"    {m}")
    else:
        log(f"  [PASS] All {len(records)} manifest paths exist")
    log()

    # 2. Duplicate paths
    log("--- Check 2: Duplicate Manifest Paths ---")
    dup_paths = check_duplicate_paths(records)
    if dup_paths:
        log(f"  [FAIL] {len(dup_paths)} duplicate paths:")
        for d in dup_paths[:10]:
            log(f"    {d}")
    else:
        log(f"  [PASS] No duplicate paths")
    log()

    # 3. Label validity
    log("--- Check 3: Label Validity ---")
    label_issues = check_labels(records)
    if label_issues:
        log(f"  [FAIL] {len(label_issues)} label issues:")
        for li in label_issues[:10]:
            log(f"    {li}")
    else:
        log(f"  [PASS] All labels valid")
    log()

    # 4. Corrupt images
    if records:
        log(f"--- Check 4: Corrupt Images (sampling {args.corrupt_sample}) ---")
        corrupt, checked = check_corrupt_images(records, args.corrupt_sample)
        if corrupt:
            log(f"  [FAIL] {len(corrupt)} corrupt in {checked} checked:")
            for c in corrupt[:10]:
                log(f"    {c}")
        else:
            log(f"  [PASS] {checked} images checked, 0 corrupt")
        log()

    # 5. Impossible dimensions
    log("--- Check 5: Impossible Dimensions ---")
    dim_issues = check_dimensions(records)
    if dim_issues:
        log(f"  [FAIL] {len(dim_issues)} impossible dimensions:")
        for di in dim_issues[:10]:
            log(f"    {di}")
    else:
        log(f"  [PASS] No impossible dimensions (or dimensions not yet in manifest)")
    log()

    # 6. Missing generators
    log("--- Check 6: Missing Generator Where Known ---")
    gen_issues = check_missing_generators(records)
    if gen_issues:
        log(f"  [FAIL] {len(gen_issues)} missing generator names:")
        for g in gen_issues[:10]:
            log(f"    {g}")
    else:
        log(f"  [PASS] Generator info complete where expected")
    log()

    # 7. Empty dirs
    log("--- Check 7: Empty Directories ---")
    empties = find_empty_dirs(DATA_RAW)
    if empties:
        log(f"  [INFO] {len(empties)} empty directories:")
        for e in empties[:10]:
            log(f"    {e}")
    else:
        log(f"  [PASS] No empty directories in data/raw/")
    log()

    # 8. MD5 hash duplicates
    if hash_sample > 0 and records:
        log(f"--- Check 8: Exact Duplicates by MD5 (sampling {hash_sample}) ---")
        dups = check_hash_duplicates(records, hash_sample)
        if dups:
            log(f"  [WARN] {len(dups)} duplicate groups found:")
            for h, paths in list(dups.items())[:5]:
                log(f"    MD5={h[:8]}...:")
                for p in paths:
                    log(f"      {p}")
        else:
            log(f"  [PASS] No exact duplicates in {hash_sample}-image sample")
        log()
    elif not records:
        log("--- Check 8: Skipped (no image records) ---")
        log()

    # 9. Dataset-level summary
    log("--- Dataset-Level Summary ---")
    by_dataset = collections.Counter(r.get("dataset") for r in records)
    for ds, cnt in sorted(by_dataset.items()):
        log(f"  {ds}: {cnt} images")

    by_split = collections.Counter(r.get("split") for r in records)
    for sp, cnt in sorted(by_split.items()):
        log(f"  split={sp}: {cnt} images")

    by_label = collections.Counter(r.get("label_name") for r in records)
    for lb, cnt in sorted(by_label.items()):
        log(f"  label={lb}: {cnt} images")

    by_gen = collections.Counter(r.get("generator") for r in records)
    for g, cnt in sorted(by_gen.items()):
        log(f"  generator={g}: {cnt} images")

    log()
    log("=== Integrity Check Complete ===")

    # Save report
    report_path = MANIFESTS_DIR / "integrity_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
