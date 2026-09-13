"""
tests/conftest.py — shared fixtures for all tests.

Sets up sys.path and provides a `torch_available` fixture / skip marker
so tests that need torch are correctly skipped when a system Application
Control policy blocks torch DLLs (e.g., in certain corporate environments).

Production training/inference always requires a real torch install.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent

# Add project root to sys.path (idempotent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


TORCH_AVAILABLE = _check_torch()

# Convenience skip marker — import in any test file
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason=(
        "PyTorch not available — DLLs may be blocked by Application Control policy. "
        "Run on Colab/unconstrained machine for full testing."
    ),
)
