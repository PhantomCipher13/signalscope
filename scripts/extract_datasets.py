#!/usr/bin/env python3
"""
scripts/extract_datasets.py
=============================
Safely extract the four SignalScope dataset/repository ZIPs into data/raw/.

Safety checks:
  - Path traversal detection
  - Duplicate extraction directory detection
  - Integrity check before extraction
  - Skips extraction if target already fully extracted

Usage:
    python scripts/extract_datasets.py
    python scripts/extract_datasets.py --force  # re-extract even if already done

IMPORTANT: Original ZIP files are NEVER deleted or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Mapping: zip_path -> extraction subdirectory name
ZIP_CONFIGS = [
    {
        "zip": PROJECT_ROOT.parent / "CIFAKE-Real-and-AI-Generated-Synthetic-Images-main.zip",
        "subdir": "CIFAKE",
        "strip_prefix": "CIFAKE-Real-and-AI-Generated-Synthetic-Images-main/",
        "description": "CIFAKE: Real and AI-Generated Synthetic Images",
    },
    {
        "zip": PROJECT_ROOT.parent / "GenImage-main.zip",
        "subdir": "GenImage",
        "strip_prefix": "GenImage-main/",
        "description": "GenImage benchmark (code + detector implementations)",
    },
    {
        "zip": PROJECT_ROOT.parent / "RAID-main.zip",
        "subdir": "RAID",
        "strip_prefix": "RAID-main/",
        "description": "RAID: Robust AI-generated Image Detection (code only)",
    },
    {
        "zip": PROJECT_ROOT.parent / "UnbiasedGenImage-master.zip",
        "subdir": "UnbiasedGenImage",
        "strip_prefix": "UnbiasedGenImage-master/",
        "description": "UnbiasedGenImage (code, metadata tools, results)",
    },
]


def check_path_traversal(zip_path: Path) -> list[str]:
    """Return a list of suspicious paths (path traversal attempts)."""
    suspicious = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name.split("/"):
                suspicious.append(name)
    return suspicious


def safe_extract(zip_path: Path, target_dir: Path, strip_prefix: str, force: bool = False) -> dict:
    """Extract zip_path into target_dir, stripping strip_prefix from paths."""
    target_dir.mkdir(parents=True, exist_ok=True)

    sentinel = target_dir / ".extracted_ok"
    if sentinel.exists() and not force:
        print(f"  [SKIP] Already extracted (delete {sentinel} to re-extract)")
        with zipfile.ZipFile(zip_path) as zf:
            n_files = sum(1 for n in zf.namelist() if not n.endswith("/"))
        return {"status": "skipped", "n_files": n_files}

    extracted = 0
    skipped = 0
    errors = []

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for member in members:
            name = member.filename

            # Strip prefix
            if strip_prefix and name.startswith(strip_prefix):
                relative = name[len(strip_prefix):]
            else:
                relative = name

            if not relative or relative.endswith("/"):
                continue  # directory entry

            out_path = target_dir / relative

            # Safety: ensure we don't escape the target directory
            try:
                out_path.resolve().relative_to(target_dir.resolve())
            except ValueError:
                errors.append(f"Path traversal blocked: {name}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
            except Exception as e:
                errors.append(f"Error extracting {name}: {e}")

    sentinel.write_text("ok")
    return {
        "status": "extracted",
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
    }


def verify_zip(zip_path: Path) -> bool:
    """Return True if ZIP CRC checks pass."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            result = zf.testzip()
            if result is not None:
                print(f"  [ERROR] Corrupt member: {result}")
                return False
        return True
    except zipfile.BadZipFile as e:
        print(f"  [ERROR] Bad ZIP: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SignalScope datasets.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already done.")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Extraction root: {DATA_RAW}\n")

    any_error = False
    for cfg in ZIP_CONFIGS:
        zip_path = cfg["zip"]
        subdir = DATA_RAW / cfg["subdir"]
        strip = cfg["strip_prefix"]
        desc = cfg["description"]

        print(f"=== {cfg['subdir']} ===")
        print(f"  Description: {desc}")
        print(f"  ZIP:         {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Target:      {subdir}")

        if not zip_path.exists():
            print(f"  [ERROR] ZIP not found: {zip_path}")
            any_error = True
            continue

        # Path traversal check
        suspicious = check_path_traversal(zip_path)
        if suspicious:
            print(f"  [ERROR] Path traversal detected in {zip_path.name}:")
            for s in suspicious[:5]:
                print(f"    {s}")
            any_error = True
            continue
        print(f"  [OK] No path traversal issues")

        # Integrity check
        print(f"  Verifying ZIP integrity...")
        if not verify_zip(zip_path):
            any_error = True
            continue
        print(f"  [OK] ZIP integrity OK")

        # Extract
        print(f"  Extracting...")
        result = safe_extract(zip_path, subdir, strip, force=args.force)
        if result["status"] == "skipped":
            print(f"  [OK] Already extracted ({result['n_files']} files)")
        elif result["errors"]:
            print(f"  [WARN] Extracted {result['extracted']} files with {len(result['errors'])} errors:")
            for e in result["errors"][:5]:
                print(f"    {e}")
        else:
            print(f"  [OK] Extracted {result['extracted']} files")

        print()

    if any_error:
        print("[FAILED] One or more ZIPs had errors. See above.")
        sys.exit(1)
    else:
        print("[OK] All datasets extracted successfully.")
        print(f"Location: {DATA_RAW}")


if __name__ == "__main__":
    main()
