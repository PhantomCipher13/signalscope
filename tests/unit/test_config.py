"""
Unit tests — Configuration system.

Tests that:
- YAML config files load without errors.
- Expected top-level sections exist in each config.
- The config loader raises clear errors on bad inputs.
- reset_config() enables a clean re-load between tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is on the path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, load_yaml, reset_config  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Reset the global config cache before every test."""
    reset_config()
    yield
    reset_config()


CONFIGS_DIR = PROJECT_ROOT / "configs"


# ── Individual YAML file tests ─────────────────────────────────────────────────

class TestLoadYaml:
    def test_default_yaml_loads(self):
        data = load_yaml(CONFIGS_DIR / "default.yaml")
        assert isinstance(data, dict)

    def test_model_yaml_loads(self):
        data = load_yaml(CONFIGS_DIR / "model.yaml")
        assert isinstance(data, dict)

    def test_augmentation_yaml_loads(self):
        data = load_yaml(CONFIGS_DIR / "augmentation.yaml")
        assert isinstance(data, dict)

    def test_reliability_yaml_loads(self):
        data = load_yaml(CONFIGS_DIR / "reliability.yaml")
        assert isinstance(data, dict)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [\n  unclosed list\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_yaml(bad)

    def test_empty_yaml_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("# just a comment\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_yaml(empty)


# ── Merged config tests ────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_config_loads_successfully(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_default_sections_present(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        for section in ("app", "logging", "seed", "paths"):
            assert section in cfg, f"Missing required section: '{section}'"

    def test_app_section_has_name(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        assert "name" in cfg["app"]
        assert cfg["app"]["name"] == "SignalScope"

    def test_logging_section_has_level(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        assert "level" in cfg["logging"]

    def test_seed_section_has_value(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        assert "value" in cfg["seed"]
        assert isinstance(cfg["seed"]["value"], int)

    def test_model_section_merged(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        # model.yaml should contribute a 'model' key
        assert "model" in cfg

    def test_reliability_section_merged(self):
        cfg = load_config(config_dir=CONFIGS_DIR)
        assert "reliability" in cfg

    def test_config_is_cached_on_second_call(self):
        cfg1 = load_config(config_dir=CONFIGS_DIR)
        cfg2 = load_config(config_dir=CONFIGS_DIR)
        assert cfg1 is cfg2, "Config should be cached — same object expected"

    def test_missing_config_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(config_dir=tmp_path / "no_such_dir")
