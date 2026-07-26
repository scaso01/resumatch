"""Tests for resumatch.config module."""

import pytest
from unittest.mock import patch

from resumatch.config import (
    ConfigError,
    load_config,
    validate_config,
    CONFIG,
    BASE_DIR,
    logger,
)


class TestConfigLoading:
    def test_config_loads_successfully(self):
        assert CONFIG is not None
        assert isinstance(CONFIG, dict)

    def test_config_has_scoring(self):
        assert "scoring" in CONFIG

    def test_config_has_matching(self):
        assert "matching" in CONFIG

    def test_config_has_llm(self):
        assert "llm" in CONFIG

    def test_config_has_api(self):
        assert "api" in CONFIG

    def test_base_dir_is_project_root(self):
        assert (BASE_DIR / "pyproject.toml").exists()

    def test_logger_exists(self):
        assert logger is not None
        assert logger.name == "resumatch"


class TestConfigValidation:
    def test_valid_config_passes(self):
        assert validate_config(CONFIG) is True

    def test_missing_scoring_raises(self):
        with pytest.raises(ConfigError, match="Missing required config keys"):
            validate_config({"matching": {}, "llm": {}, "api": {}})

    def test_missing_matching_raises(self):
        with pytest.raises(ConfigError, match="Missing required config keys"):
            validate_config({"scoring": {}, "llm": {}, "api": {}})

    def test_missing_llm_raises(self):
        with pytest.raises(ConfigError, match="Missing required config keys"):
            validate_config({"scoring": {}, "matching": {}, "api": {}})

    def test_missing_api_raises(self):
        with pytest.raises(ConfigError, match="Missing required config keys"):
            validate_config({"scoring": {}, "matching": {}, "llm": {}})

    def test_weights_must_sum_to_100(self):
        cfg = {
            "scoring": {"weights": {"impact": 50, "presentation": 30, "competencies": 30}},
            "matching": {},
            "llm": {},
            "api": {},
        }
        with pytest.raises(ConfigError, match="must sum to 100"):
            validate_config(cfg)

    def test_weights_exactly_100_passes(self):
        cfg = {
            "scoring": {"weights": {"impact": 40, "presentation": 30, "competencies": 30}, "zones": {}},
            "matching": {},
            "llm": {},
            "api": {},
        }
        assert validate_config(cfg) is True

    def test_green_must_be_above_yellow(self):
        cfg = {
            "scoring": {
                "weights": {"impact": 40, "presentation": 30, "competencies": 30},
                "zones": {"green": 50, "yellow": 85},
            },
            "matching": {},
            "llm": {},
            "api": {},
        }
        with pytest.raises(ConfigError, match="Green zone.*must be > yellow"):
            validate_config(cfg)

    def test_missing_weight_keys_raises(self):
        cfg = {
            "scoring": {"weights": {"impact": 50, "presentation": 50}},
            "matching": {},
            "llm": {},
            "api": {},
        }
        with pytest.raises(ConfigError, match="Missing scoring weight keys"):
            validate_config(cfg)

    def test_empty_config_raises(self):
        with pytest.raises(ConfigError):
            validate_config({})


class TestLoadConfig:
    def test_load_config_missing_file(self, tmp_path):
        with patch("resumatch.config.CONFIG_DIR", tmp_path):
            with pytest.raises(ConfigError, match="config.yaml not found"):
                load_config()

    def test_load_config_invalid_yaml(self, tmp_path):
        bad_yaml = tmp_path / "config.yaml"
        bad_yaml.write_text("{{invalid: yaml: [", encoding="utf-8")
        with patch("resumatch.config.CONFIG_DIR", tmp_path):
            with pytest.raises(ConfigError, match="invalid YAML"):
                load_config()

    def test_load_config_empty_file(self, tmp_path):
        empty_yaml = tmp_path / "config.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        with patch("resumatch.config.CONFIG_DIR", tmp_path):
            result = load_config()
            assert result == {}


class TestConfigValues:
    def test_scoring_weights_sum_to_100(self):
        weights = CONFIG["scoring"]["weights"]
        assert sum(weights.values()) == 100

    def test_scoring_zones_ordered(self):
        zones = CONFIG["scoring"]["zones"]
        assert zones["green"] > zones["yellow"]

    def test_llm_has_base_url(self):
        assert "base_url" in CONFIG["llm"]

    def test_api_has_port(self):
        assert "port" in CONFIG["api"]
        assert isinstance(CONFIG["api"]["port"], int)
