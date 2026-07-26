"""
ResuMatch - Configuration Module
==================================
Centralized configuration, logging, and shared constants.

All other ResuMatch modules import from here:
    from resumatch.config import CONFIG, BASE_DIR, logger
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("RESUMATCH_DATA_DIR", str(BASE_DIR)))
CONFIG_DIR = Path(os.environ.get("RESUMATCH_CONFIG_DIR", str(BASE_DIR)))

load_dotenv(CONFIG_DIR / ".env")


def setup_logging() -> logging.Logger:
    """Configure logging with file rotation and console output."""
    _logger = logging.getLogger("resumatch")
    _logger.setLevel(logging.DEBUG)

    if _logger.handlers:
        return _logger

    log_file = DATA_DIR / "resumatch.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    return _logger


logger = setup_logging()


ENV_PREFIX = "RESUMATCH_"
ENV_NESTED_SEPARATOR = "__"


def _coerce(raw: str, current: Any) -> Any:
    """Parse an env string to match the type of the value it replaces."""
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Override config values from RESUMATCH_<SECTION>__<KEY> env vars.

    Lets a deployment change any setting without editing config.yaml, e.g.
    RESUMATCH_LLM__BASE_URL=http://192.0.2.10:8080
    RESUMATCH_LLM__ENABLED=false
    RESUMATCH_API__CORS_ORIGINS=http://localhost:8511,http://localhost:3000

    Only keys already present in config.yaml can be overridden, so a typo
    fails loudly at startup instead of silently doing nothing.
    """
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX):].lower().split(ENV_NESTED_SEPARATOR)
        if len(path) < 2:
            continue  # RESUMATCH_DATA_DIR and friends are not config keys
        section, *rest = path
        if section not in config or not isinstance(config[section], dict):
            continue
        node: Any = config[section]
        for part in rest[:-1]:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        leaf = rest[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(
                f"{env_key} does not match any setting in config.yaml"
            )
        node[leaf] = _coerce(raw, node[leaf])
    return config


def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml, then apply env overrides.

    Raises:
        ConfigError: If config.yaml is missing or contains invalid YAML.
    """
    config_path = CONFIG_DIR / "config.yaml"
    if not config_path.exists():
        raise ConfigError(f"config.yaml not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            loaded = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"config.yaml contains invalid YAML: {e}") from e
    return apply_env_overrides(loaded)


def validate_config(config: dict[str, Any]) -> bool:
    """Validate config has required keys. Returns True if valid.

    Raises:
        ConfigError: If required configuration keys are missing or invalid.
    """
    required_top = ["scoring", "matching", "llm", "api"]
    missing = [k for k in required_top if k not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    scoring = config.get("scoring", {})
    weights = scoring.get("weights", {})
    if weights:
        weight_sum = sum(weights.values())
        if weight_sum != 100:
            raise ConfigError(
                f"Scoring weights must sum to 100, got {weight_sum}. "
                f"Current weights: {weights}"
            )

    required_weight_keys = ["impact", "presentation", "competencies"]
    missing_weights = [k for k in required_weight_keys if k not in weights]
    if missing_weights:
        raise ConfigError(f"Missing scoring weight keys: {', '.join(missing_weights)}")

    zones = scoring.get("zones", {})
    if zones:
        green = zones.get("green", 85)
        yellow = zones.get("yellow", 50)
        if green <= yellow:
            raise ConfigError(f"Green zone ({green}) must be > yellow zone ({yellow})")

    return True


CONFIG = load_config()
validate_config(CONFIG)
