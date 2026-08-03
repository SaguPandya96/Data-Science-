from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {"data", "validation", "features", "model", "scenarios", "paths"}


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML configuration and return it with the resolved project root."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping at the top level.")

    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    project_root = (path.parent / config.get("project_root", ".")).resolve()
    return config, project_root


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a configured project-relative path without requiring it to exist."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()
