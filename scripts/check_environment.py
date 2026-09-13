#!/usr/bin/env python3
"""
SignalScope — scripts/check_environment.py
===========================================
Verifies that the development environment satisfies Phase 0 + Phase 1 requirements.

Run from the project root:
    python scripts/check_environment.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed

PYTHON: C:\\Users\\Admin\\AppData\\Local\\python-embed\\python.exe
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ── Minimum Python version ────────────────────────────────────────────────────
REQUIRED_PYTHON = (3, 9)

# ── Packages required for Phase 0 + Phase 1 ──────────────────────────────────
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("yaml", "pyyaml"),
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
    ("pytest", "pytest"),
    # Phase 1 ML packages
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
]

# ── Optional packages (Phase 2+) — warn only, do not fail ────────────────────
OPTIONAL_PACKAGES: list[tuple[str, str]] = [
    ("netcal", "netcal"),          # Phase 2: calibration/ECE
    ("piexif", "piexif"),          # Phase 2: EXIF reading
]

# ── Required project directories ──────────────────────────────────────────────
REQUIRED_DIRS: list[str] = [
    "app",
    "src",
    "configs",
    "models",
    "data/splits",
    "scripts",
    "experiments",
    "tests/unit",
    "tests/integration",
    "logs",
    "outputs",
    "frontend",
    "docs",
]

# ── Required config files ─────────────────────────────────────────────────────
REQUIRED_CONFIGS: list[str] = [
    "configs/default.yaml",
    "configs/model.yaml",
    "configs/augmentation.yaml",
    "configs/reliability.yaml",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}[PASS]{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 50)


# ── Check functions ───────────────────────────────────────────────────────────

def check_python_version() -> bool:
    _section("Python Version")
    ver = sys.version_info[:2]
    required = REQUIRED_PYTHON
    ok = ver >= required
    status = f"Python {ver[0]}.{ver[1]} (required >= {required[0]}.{required[1]})"
    if ok:
        _ok(status)
    else:
        _fail(status)
    return ok


def check_required_packages() -> bool:
    _section("Required Packages (Phase 0)")
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            _ok(f"{pip_name} {version}")
        except ImportError:
            _fail(f"{pip_name} — NOT INSTALLED  (pip install {pip_name})")
            all_ok = False
    return all_ok


def check_optional_packages() -> None:
    _section("Optional Packages (Phase 1+) — warnings only")
    for import_name, pip_name in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            _ok(f"{pip_name} {version}")
        except ImportError:
            _warn(f"{pip_name} — not installed (needed for Phase 1+)")


def check_directories(project_root: Path) -> bool:
    _section("Project Directories")
    all_ok = True
    for d in REQUIRED_DIRS:
        path = project_root / d
        if path.is_dir():
            _ok(str(d))
        else:
            _fail(f"{d}  — directory missing")
            all_ok = False
    return all_ok


def check_config_files(project_root: Path) -> bool:
    _section("Configuration Files")
    all_ok = True
    for cfg in REQUIRED_CONFIGS:
        path = project_root / cfg
        if path.is_file():
            _ok(cfg)
        else:
            _fail(f"{cfg}  — file missing")
            all_ok = False
    return all_ok


def check_config_loading(project_root: Path) -> bool:
    _section("Configuration Loading")
    # Add project root and src to path so imports work.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from src.utils import load_config, reset_config  # noqa: PLC0415
        reset_config()  # ensure fresh load
        cfg = load_config(config_dir=project_root / "configs")
        required_sections = ["app", "logging", "seed", "paths"]
        missing = [s for s in required_sections if s not in cfg]
        if missing:
            _fail(f"Configuration loaded but missing sections: {missing}")
            return False
        _ok(f"Configuration loaded successfully ({len(cfg)} top-level sections)")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(f"Configuration loading failed: {exc}")
        return False


def check_src_imports(project_root: Path) -> bool:
    _section("Source Module Imports")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    modules = [
        "src.utils",
        "src.detector",
        "src.preprocessing",
        "src.dataset",       # Phase 1
        "src.evaluation",    # Phase 1
        "src.calibration",
        "src.reliability",
        "src.gradcam",
        "src.provenance",
        "src.formatter",
    ]
    all_ok = True
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            _ok(mod_name)
        except ImportError as exc:
            _fail(f"{mod_name}  - {exc}")
            all_ok = False
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}=== SignalScope - Environment Check ==={RESET}")

    # Resolve project root (one level above this script's directory).
    project_root = Path(__file__).resolve().parent.parent
    print(f"  Project root: {project_root}")

    # Change working directory to project root so relative paths work.
    os.chdir(project_root)

    results: list[bool] = [
        check_python_version(),
        check_required_packages(),
        check_directories(project_root),
        check_config_files(project_root),
        check_config_loading(project_root),
        check_src_imports(project_root),
    ]

    # Optional (never fails the overall check)
    check_optional_packages()

    print()
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"{BOLD}{GREEN}== ENVIRONMENT CHECK PASSED ({passed}/{total}) =={RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}== ENVIRONMENT CHECK FAILED ({passed}/{total} passed) =={RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
