"""Scenario persistence.

Suites are stored as JSONL — one scenario per line — rather than a single JSON document.
The reason is practical: a 150-scenario suite regenerated with a changed template
produces a readable line-level diff in JSONL and an unreadable one in pretty-printed
JSON, and reviewing what a generator change actually did is a thing that happens often.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalforge.exceptions import ScenarioError
from evalforge.ids import stable_id
from evalforge.logging_config import get_logger
from evalforge.schemas.scenario import Scenario, ScenarioSuite

logger = get_logger(__name__)


def save_scenarios(scenarios: list[Scenario], path: Path) -> Path:
    """Write scenarios to a JSONL file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(scenario.model_dump_json())
            handle.write("\n")
    logger.info("scenarios_saved", path=str(path), count=len(scenarios))
    return path


def load_scenarios(path: Path) -> list[Scenario]:
    """Read scenarios from a JSONL file.

    Raises:
        ScenarioError: If the file is missing or any line fails validation.
    """
    if not path.exists():
        raise ScenarioError(
            f"Scenario file not found: {path}. Generate one with `evalforge generate`."
        )
    scenarios: list[Scenario] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                scenarios.append(Scenario.model_validate_json(stripped))
            except Exception as exc:
                raise ScenarioError(f"{path}:{line_number} is not a valid scenario: {exc}") from exc
    return scenarios


def save_suite(suite: ScenarioSuite, directory: Path) -> Path:
    """Write a suite as scenarios plus a manifest.

    The manifest records seed, generator version and composition, so a stored suite can
    be regenerated exactly without guessing which arguments produced it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    scenarios_path = directory / f"{suite.name}_scenarios.jsonl"
    save_scenarios(suite.scenarios, scenarios_path)

    manifest = {
        "suite_id": suite.suite_id,
        "name": suite.name,
        "description": suite.description,
        "generator_version": suite.generator_version,
        "seed": suite.seed,
        "scenario_count": len(suite.scenarios),
        "scenarios_file": scenarios_path.name,
        "categories": sorted({s.category.value for s in suite.scenarios}),
        "difficulties": sorted({s.difficulty.value for s in suite.scenarios}),
        "lengths": sorted({s.turn_count for s in suite.scenarios}),
    }
    manifest_path = directory / f"{suite.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scenarios_path


def load_suite(directory: Path, name: str = "core") -> ScenarioSuite:
    """Read a suite written by :func:`save_suite`.

    Raises:
        ScenarioError: If neither the manifest nor the scenario file is present.
    """
    manifest_path = directory / f"{name}_manifest.json"
    scenarios_path = directory / f"{name}_scenarios.jsonl"

    if not scenarios_path.exists():
        raise ScenarioError(
            f"No suite named {name!r} in {directory}. "
            f"Generate one with `evalforge generate --suite {name}`."
        )

    scenarios = load_scenarios(scenarios_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ScenarioSuite(
            suite_id=manifest.get("suite_id", stable_id("scenario", "suite", name)),
            name=manifest.get("name", name),
            description=manifest.get("description", ""),
            generator_version=manifest.get("generator_version", "unknown"),
            seed=int(manifest.get("seed", 0)),
            scenarios=scenarios,
        )

    return ScenarioSuite(
        suite_id=stable_id("scenario", "suite", name),
        name=name,
        scenarios=scenarios,
    )


def suite_exists(directory: Path, name: str = "core") -> bool:
    """Whether a suite with this name has been generated."""
    return (directory / f"{name}_scenarios.jsonl").exists()
