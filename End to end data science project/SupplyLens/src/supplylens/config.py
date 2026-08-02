"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = {
    "project",
    "data",
    "target",
    "splits",
    "modeling",
    "operations",
    "outputs",
    "monitoring",
}


def project_root(start: Path | None = None) -> Path:
    """Return the repository root by locating ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root containing pyproject.toml")


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration and validate required sections and key values."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    threshold = config["target"].get("delay_days_threshold")
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("target.delay_days_threshold must be a non-negative integer")

    capacities = config["operations"].get("review_capacity_levels", [])
    if not capacities or any(not 0 < float(value) <= 1 for value in capacities):
        raise ValueError("Review capacities must be numbers in the interval (0, 1]")

    required_paths = {"raw_path", "processed_path", "sha256"}
    missing_paths = required_paths - set(config["data"])
    if missing_paths:
        raise ValueError(f"data section is missing: {sorted(missing_paths)}")
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else project_root() / path

