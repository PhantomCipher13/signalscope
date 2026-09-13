"""
Unit tests — Utilities: logging and seeding.

Tests that:
- The logger initialises under the 'signalscope' namespace.
- Child loggers are correctly namespaced.
- The seed utility runs without errors.
- seed_from_config reads the seed value from config.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    LOGGER_NAME,
    get_logger,
    reset_config,
    seed_from_config,
    set_seed,
    setup_logging,
)

# ── Reset logging state between tests ────────────────────────────────────────
import src.utils as _utils_module


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset logging and config state before every test."""
    _utils_module._logging_initialised = False
    # Remove all handlers from the signalscope logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    reset_config()
    yield
    _utils_module._logging_initialised = False
    logger.handlers.clear()
    reset_config()


# ── Logger tests ──────────────────────────────────────────────────────────────

class TestLogging:
    def test_setup_logging_returns_logger(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)
        assert isinstance(logger, logging.Logger)

    def test_logger_name_is_signalscope(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)
        assert logger.name == LOGGER_NAME

    def test_logger_has_handlers_after_setup(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)
        assert len(logger.handlers) > 0

    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "nested" / "logs"
        setup_logging(log_dir=log_dir)
        assert log_dir.is_dir()

    def test_second_call_is_noop(self, tmp_path):
        logger1 = setup_logging(log_dir=tmp_path)
        n_handlers = len(logger1.handlers)
        logger2 = setup_logging(log_dir=tmp_path)
        # Should not add more handlers
        assert len(logger2.handlers) == n_handlers

    def test_get_logger_returns_child_logger(self, tmp_path):
        setup_logging(log_dir=tmp_path)
        child = get_logger("detector")
        assert child.name == f"{LOGGER_NAME}.detector"

    def test_get_logger_root_without_name(self, tmp_path):
        setup_logging(log_dir=tmp_path)
        root = get_logger()
        assert root.name == LOGGER_NAME

    def test_logger_does_not_propagate_image_bytes(self, tmp_path, caplog):
        """Ensure logging at INFO level emits a string, not binary data."""
        setup_logging(log_dir=tmp_path)
        logger = get_logger("test")
        fake_message = "Processing image file: example.jpg (size: 1024x768)"
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            logger.info(fake_message)
        # The log message should be the string, not raw bytes
        assert any(fake_message in r.message for r in caplog.records)


# ── Seed tests ────────────────────────────────────────────────────────────────

class TestSeeding:
    def test_set_seed_runs_without_error(self):
        # Should not raise regardless of whether torch is installed.
        set_seed(42)

    def test_set_seed_affects_python_random(self):
        set_seed(123)
        val1 = random.random()
        set_seed(123)
        val2 = random.random()
        assert val1 == val2, "Same seed should produce same random value"

    def test_different_seeds_produce_different_values(self):
        set_seed(1)
        val1 = random.random()
        set_seed(2)
        val2 = random.random()
        assert val1 != val2

    def test_seed_from_config_uses_config_value(self, tmp_path):
        """seed_from_config should read seed.value from the config dict."""
        cfg = {
            "seed": {"value": 77, "deterministic": False},
            "logging": {"log_dir": str(tmp_path)},
        }
        seed_from_config(cfg)
        val1 = random.random()

        set_seed(77)
        val2 = random.random()
        assert val1 == val2, "seed_from_config should apply the configured seed"

    def test_numpy_seeded_when_available(self):
        """NumPy is required in Phase 1 — seeding must be consistent."""
        import numpy as np  # NumPy is a required Phase 1 dependency
        set_seed(999)
        arr1 = np.random.rand(5).tolist()
        set_seed(999)
        arr2 = np.random.rand(5).tolist()
        assert arr1 == arr2
