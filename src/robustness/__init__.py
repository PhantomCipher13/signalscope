"""
SignalScope — src/robustness/__init__.py
==========================================
Robustness probe framework.

Exports the probe runner for applying deterministic image transformations
and evaluating model stability under each transformation.
"""
from .probes import (
    ProbeConfig,
    ProbeResult,
    ProbeRunner,
    BUILTIN_PROBES,
)
from .stress_test import (
    StressTestConfig,
    StressTestResult,
    StressTestEngine,
)

__all__ = [
    "ProbeConfig",
    "ProbeResult",
    "ProbeRunner",
    "BUILTIN_PROBES",
    "StressTestConfig",
    "StressTestResult",
    "StressTestEngine",
]
