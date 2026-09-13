"""
SignalScope — src/torch_compat.py
===================================
Thin compatibility shim that lets non-torch code paths run safely when
torch DLLs are blocked by a system Application Control policy.

Usage in source modules (lazy import pattern):
    from src.torch_compat import torch_available, require_torch
    if torch_available():
        import torch

This shim is NOT a replacement for torch. It only allows:
  1. Code that optionally uses torch to degrade gracefully.
  2. Tests that mock torch to run without physical GPU/DLL access.

Production training and inference always require a working torch install.
"""
from __future__ import annotations

import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def torch_available() -> bool:
    """Return True if torch can be imported successfully."""
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def require_torch(context: str = "") -> None:
    """Raise ImportError with a helpful message if torch is unavailable."""
    if not torch_available():
        msg = (
            "PyTorch is required but could not be loaded. "
            "Possible causes:\n"
            "  1. torch is not installed.\n"
            "  2. Windows Application Control policy is blocking torch DLLs.\n"
            "     Solution: run training on Google Colab (T4 GPU) or a machine\n"
            "     without WDAC restrictions.\n"
            "     See docs/PHASE1_NOTES.md for instructions.\n"
        )
        if context:
            msg = f"[{context}] " + msg
        raise ImportError(msg)
