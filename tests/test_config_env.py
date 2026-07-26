"""Tests for RESUMATCH_<SECTION>__<KEY> environment overrides.

This is what makes config.yaml deployable without editing it -- and what makes
.env.example honest, since before this every variable it documented was read by
nothing at all.
"""

from __future__ import annotations

import pytest

from resumatch.config import ConfigError, apply_env_overrides


@pytest.fixture
def config() -> dict:
    """A miniature stand-in for config.yaml with one of each value type."""
    return {
        "llm": {
            "base_url": "http://localhost:8080",
            "enabled": True,
            "timeout_seconds": 120,
            "temperature": 0.3,
        },
        "api": {
            "port": 8510,
            "cors_origins": ["http://localhost:8511"],
        },
        "scoring": {"zones": {"green": 85, "yellow": 50}},
    }


def test_no_env_vars_leaves_config_untouched(monkeypatch, config):
    monkeypatch.delenv("RESUMATCH_LLM__BASE_URL", raising=False)
    before = {"llm": dict(config["llm"]), "api": dict(config["api"])}
    apply_env_overrides(config)
    assert config["llm"] == before["llm"]
    assert config["api"] == before["api"]


def test_overrides_a_string(monkeypatch, config):
    monkeypatch.setenv("RESUMATCH_LLM__BASE_URL", "http://192.0.2.10:9000")
    apply_env_overrides(config)
    assert config["llm"]["base_url"] == "http://192.0.2.10:9000"


@pytest.mark.parametrize(
    "raw,expected",
    [("false", False), ("0", False), ("no", False), ("true", True), ("1", True)],
)
def test_overrides_a_bool(monkeypatch, config, raw, expected):
    monkeypatch.setenv("RESUMATCH_LLM__ENABLED", raw)
    apply_env_overrides(config)
    assert config["llm"]["enabled"] is expected


def test_overrides_an_int(monkeypatch, config):
    monkeypatch.setenv("RESUMATCH_API__PORT", "9999")
    apply_env_overrides(config)
    assert config["api"]["port"] == 9999
    assert isinstance(config["api"]["port"], int)


def test_overrides_a_float(monkeypatch, config):
    monkeypatch.setenv("RESUMATCH_LLM__TEMPERATURE", "0.9")
    apply_env_overrides(config)
    assert config["llm"]["temperature"] == pytest.approx(0.9)


def test_overrides_a_list_from_csv(monkeypatch, config):
    monkeypatch.setenv(
        "RESUMATCH_API__CORS_ORIGINS", "http://a:1, http://b:2 ,http://c:3"
    )
    apply_env_overrides(config)
    assert config["api"]["cors_origins"] == [
        "http://a:1",
        "http://b:2",
        "http://c:3",
    ]


def test_overrides_a_nested_key(monkeypatch, config):
    monkeypatch.setenv("RESUMATCH_SCORING__ZONES__GREEN", "90")
    apply_env_overrides(config)
    assert config["scoring"]["zones"]["green"] == 90


def test_unknown_key_fails_loudly(monkeypatch, config):
    """A typo must not silently do nothing -- that was the old failure mode."""
    monkeypatch.setenv("RESUMATCH_LLM__BSAE_URL", "http://typo")
    with pytest.raises(ConfigError, match="RESUMATCH_LLM__BSAE_URL"):
        apply_env_overrides(config)


def test_path_vars_are_not_treated_as_config_keys(monkeypatch, config):
    """RESUMATCH_DATA_DIR has no '__', so it must be ignored, not rejected."""
    monkeypatch.setenv("RESUMATCH_DATA_DIR", "/tmp/x")
    monkeypatch.setenv("RESUMATCH_CONFIG_DIR", "/tmp/y")
    monkeypatch.setenv("RESUMATCH_API_URL", "http://localhost:8510")
    apply_env_overrides(config)  # must not raise
    assert config["api"]["port"] == 8510


def test_unknown_section_is_ignored(monkeypatch, config):
    """Env vars aimed at sections we do not own are left alone."""
    monkeypatch.setenv("RESUMATCH_SOMETHINGELSE__FOO", "bar")
    apply_env_overrides(config)  # must not raise
    assert "somethingelse" not in config
