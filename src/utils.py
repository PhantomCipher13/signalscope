"""
SignalScope — src/utils.py
==========================
Shared utilities: configuration loading, logging, and deterministic seeding.

All other modules should obtain their logger via ``get_logger()`` and their
configuration via ``load_config()`` / ``get_config()``.  No module should
directly call ``logging.basicConfig`` or ``yaml.safe_load`` independently.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

# ── Module-level logger name (all child loggers inherit from this) ───────────
LOGGER_NAME = "signalscope"

# ── Module-level sentinel so logging is only initialised once ────────────────
_logging_initialised: bool = False

# ── Cached merged configuration ──────────────────────────────────────────────
_config: dict[str, Any] | None = None


# ============================================================
# Configuration
# ============================================================

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and return a single YAML file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file is malformed YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        # An empty YAML file is not a valid config — be explicit.
        raise ValueError(
            f"Configuration file is empty or contains only comments: {path}"
        )
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping at the top level, got "
            f"{type(data).__name__} in: {path}"
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_dir: str | Path = "configs",
    *,
    extra_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Load and merge all SignalScope configuration YAML files.

    Loading order (each file overrides previous):
    1. ``default.yaml`` — base settings
    2. ``model.yaml``
    3. ``augmentation.yaml``
    4. ``reliability.yaml``
    5. Any files listed in *extra_files*

    The merged result is cached so subsequent calls are free.

    Parameters
    ----------
    config_dir:
        Directory that contains the standard config files.
        Defaults to ``"configs"`` (relative to the working directory).
    extra_files:
        Optional additional YAML files to merge after the standard ones.

    Returns
    -------
    dict
        Merged configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``default.yaml`` (the mandatory base) is missing.
    yaml.YAMLError / ValueError / TypeError
        On malformed YAML or unexpected top-level structure.
    """
    global _config
    if _config is not None:
        return _config

    config_dir = Path(config_dir)
    standard_files = [
        config_dir / "default.yaml",
        config_dir / "model.yaml",
        config_dir / "augmentation.yaml",
        config_dir / "reliability.yaml",
    ]

    # default.yaml is mandatory
    merged: dict[str, Any] = load_yaml(standard_files[0])

    # Remaining standard files are merged if present
    for cfg_file in standard_files[1:]:
        if cfg_file.exists():
            merged = _deep_merge(merged, load_yaml(cfg_file))

    # Extra overrides
    for ef in extra_files or []:
        merged = _deep_merge(merged, load_yaml(ef))

    _config = merged
    return _config


def get_config() -> dict[str, Any]:
    """Return the cached configuration, loading defaults if not yet loaded.

    This is the preferred accessor for modules that do not need to control
    the config directory.
    """
    if _config is None:
        return load_config()
    return _config


def reset_config() -> None:
    """Clear the cached configuration (primarily for use in tests)."""
    global _config
    _config = None


# ============================================================
# Logging
# ============================================================

def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path = "logs",
    filename: str = "signalscope.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
) -> logging.Logger:
    """Initialise the ``signalscope`` logger.

    Sets up:
    - A StreamHandler writing to stderr (always).
    - A RotatingFileHandler writing to ``<log_dir>/<filename>``.

    Safe to call multiple times — subsequent calls after the first are no-ops.

    Security note: Never log raw image bytes or uploaded-file contents.

    Parameters
    ----------
    level:
        Log level string (e.g. ``"DEBUG"``) or integer constant.
    log_dir:
        Directory for log files; created automatically if absent.
    filename:
        Base filename for the rotating log.
    max_bytes:
        Maximum size of each log file before rotation.
    backup_count:
        Number of old log files to retain.
    fmt:
        Log record format string.

    Returns
    -------
    logging.Logger
        The configured root ``signalscope`` logger.
    """
    global _logging_initialised
    logger = logging.getLogger(LOGGER_NAME)

    if _logging_initialised:
        return logger

    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    logger.setLevel(numeric_level)
    formatter = logging.Formatter(fmt)

    # ── Console handler (stderr) ──────────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (rotating) ────────────────────────────────
    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(
            "Could not create file log handler (%s). "
            "Continuing with console logging only.",
            exc,
        )

    _logging_initialised = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``signalscope`` namespace.

    Initialises logging with defaults if ``setup_logging`` has not yet
    been called.

    Parameters
    ----------
    name:
        Optional sub-name (e.g. ``"detector"``).  The full logger name
        will be ``"signalscope.detector"``.  Pass ``None`` to get the
        root ``signalscope`` logger.
    """
    if not _logging_initialised:
        setup_logging()
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


# ============================================================
# Deterministic Seeding
# ============================================================

def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Trade-offs
    ----------
    Setting ``deterministic=True`` forces PyTorch to use only
    deterministic algorithms (``torch.use_deterministic_algorithms(True)``).
    On some GPU operations this can reduce throughput.  For training
    experiments prioritising speed over exact reproducibility, you may set
    ``deterministic=False`` while keeping ``torch.manual_seed`` active —
    this gives much better repeatability than no seeding, without the
    full overhead.

    Parameters
    ----------
    seed:
        Integer seed value.  Should come from ``config["seed"]["value"]``.
    deterministic:
        Whether to enable strict deterministic algorithms in PyTorch.
    """
    logger = get_logger("utils")

    # Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy (optional — imported lazily so the module is not required)
    try:
        import numpy as np  # noqa: PLC0415
        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy not installed; skipping NumPy seed.")

    # PyTorch (optional — imported lazily)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            # CuDNN deterministic mode (may reduce GPU throughput)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug(
                "PyTorch deterministic algorithms enabled. "
                "This may reduce GPU throughput on some operations."
            )
        else:
            torch.backends.cudnn.benchmark = True
            logger.debug(
                "PyTorch seeded (seed=%d) without strict determinism. "
                "Results are highly reproducible but not bit-exact.",
                seed,
            )
    except ImportError:
        logger.debug("PyTorch not installed; skipping PyTorch seed.")

    logger.debug("Random seed set to %d (deterministic=%s).", seed, deterministic)


def seed_from_config(cfg: dict[str, Any] | None = None) -> None:
    """Apply the seed defined in configuration.

    Parameters
    ----------
    cfg:
        Merged configuration dict.  If ``None``, ``get_config()`` is called.
    """
    if cfg is None:
        cfg = get_config()
    seed_cfg = cfg.get("seed", {})
    seed_value: int = seed_cfg.get("value", 42)
    deterministic: bool = seed_cfg.get("deterministic", True)
    set_seed(seed_value, deterministic=deterministic)
