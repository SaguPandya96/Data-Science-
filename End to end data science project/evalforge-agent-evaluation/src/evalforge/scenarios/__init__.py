"""Adversarial scenario generation, validation and persistence."""

from __future__ import annotations

from evalforge.scenarios.generator import (
    GENERATOR_VERSION,
    allocate_counts,
    generate_scenarios,
    generate_suite,
)
from evalforge.scenarios.library import (
    load_scenarios,
    load_suite,
    save_scenarios,
    save_suite,
    suite_exists,
)
from evalforge.scenarios.validator import (
    ScenarioReport,
    SuiteReport,
    assert_valid,
    validate_scenario,
    validate_suite,
)

__all__ = [
    "GENERATOR_VERSION",
    "ScenarioReport",
    "SuiteReport",
    "allocate_counts",
    "assert_valid",
    "generate_scenarios",
    "generate_suite",
    "load_scenarios",
    "load_suite",
    "save_scenarios",
    "save_suite",
    "suite_exists",
    "validate_scenario",
    "validate_suite",
]
