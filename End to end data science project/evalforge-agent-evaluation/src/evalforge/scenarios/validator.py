"""Scenario quality validation.

A generator can silently produce degenerate cases — a "context degradation" scenario
with no facts, an injection scenario with no injection, a suite where 90% of scenarios
share one project name. Any of those quietly weakens the evaluation while the pass rate
looks fine, so the suite is checked before it is ever executed.

Validation is split in two: **errors** make a scenario unusable, **warnings** flag
quality problems worth knowing about but not worth failing a run over.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evalforge.exceptions import ScenarioValidationError
from evalforge.schemas.common import Difficulty, InjectedFailureType, ScenarioCategory
from evalforge.schemas.scenario import ConstraintKind, Scenario, ScenarioSuite

#: Every category a complete suite must contain.
REQUIRED_CATEGORIES: frozenset[ScenarioCategory] = frozenset(ScenarioCategory)

#: Conversation lengths the length-sweep analysis expects to find.
EXPECTED_LENGTHS: tuple[int, ...] = (5, 10, 15, 20, 30)


@dataclass
class ScenarioReport:
    """Validation outcome for one scenario."""

    scenario_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the scenario is usable."""
        return not self.errors


@dataclass
class SuiteReport:
    """Validation outcome for a whole suite, plus its composition statistics."""

    total: int = 0
    valid: int = 0
    scenario_reports: list[ScenarioReport] = field(default_factory=list)
    suite_errors: list[str] = field(default_factory=list)
    suite_warnings: list[str] = field(default_factory=list)
    category_distribution: dict[str, int] = field(default_factory=dict)
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    length_distribution: dict[int, int] = field(default_factory=dict)
    tool_coverage: dict[str, int] = field(default_factory=dict)
    injected_failure_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the suite passed validation."""
        return not self.suite_errors and self.valid == self.total

    @property
    def invalid_scenarios(self) -> list[ScenarioReport]:
        """Scenarios that failed validation."""
        return [report for report in self.scenario_reports if not report.valid]


def validate_scenario(scenario: Scenario) -> ScenarioReport:
    """Check one scenario for structural and semantic problems."""
    report = ScenarioReport(scenario_id=scenario.scenario_id)

    if scenario.turn_count < 3:
        report.errors.append(f"only {scenario.turn_count} turns; multi-turn needs at least 3")

    resolved = scenario.final_fact_values()
    if not resolved:
        report.errors.append("no facts defined; nothing to test retention against")
    if not any(fact.critical for fact in resolved.values()):
        report.warnings.append("no critical facts; failures here cannot block release")

    if not scenario.expected_tool_sequence:
        report.warnings.append("no expected tool calls; tool-use dimensions will be unscored")

    unresolved = {
        cid
        for turn in scenario.conversation_turns
        for cid in (*turn.constraints_added, *turn.constraints_removed)
    } - {c.constraint_id for c in scenario.active_constraints}
    if unresolved:
        report.errors.append(f"turns reference undefined constraints: {sorted(unresolved)}")

    for constraint in scenario.active_constraints:
        if constraint.kind is ConstraintKind.MAX_WORDS and not constraint.params.get("limit"):
            report.errors.append(f"constraint {constraint.constraint_id} lacks a word limit")
        if (
            constraint.kind in {ConstraintKind.FORBID_CONTENT, ConstraintKind.REDACT_ENTITY}
            and not constraint.target
        ):
            report.errors.append(f"constraint {constraint.constraint_id} has no target to check")
        if constraint.turn_added >= scenario.turn_count:
            report.errors.append(
                f"constraint {constraint.constraint_id} activates after the last turn"
            )

    # Category-specific expectations. Each category exists to exercise one mechanism;
    # a scenario that does not contain that mechanism is mislabelled, not merely weak.
    if scenario.category is ScenarioCategory.PROMPT_INJECTION and not any(
        turn.contains_injection for turn in scenario.conversation_turns
    ):
        report.errors.append("prompt_injection scenario contains no injection payload")

    if scenario.category is ScenarioCategory.FAILURE_RECOVERY and not any(
        turn.injected_failure is not InjectedFailureType.NONE
        for turn in scenario.conversation_turns
    ):
        report.errors.append("failure_recovery scenario injects no faults")

    if scenario.category is ScenarioCategory.GOAL_DRIFT and not any(
        turn.is_distractor for turn in scenario.conversation_turns
    ):
        report.errors.append("goal_drift scenario contains no distractor turns")

    if scenario.category is ScenarioCategory.INSTRUCTION_FORGETTING and (
        len(scenario.active_constraints) < 2
    ):
        report.errors.append("instruction_forgetting scenario needs at least two constraints")

    if (
        scenario.category is ScenarioCategory.CASCADING_ERRORS
        and len(scenario.expected_tool_sequence) < 3
    ):
        report.errors.append("cascading_errors scenario needs a chain of at least three steps")

    if scenario.category is ScenarioCategory.CONTEXT_DEGRADATION:
        spans = [
            scenario.turn_count - fact.turn_introduced
            for fact in resolved.values()
            if fact.critical
        ]
        if spans and max(spans) < 3:
            report.warnings.append("critical facts are never separated from their use by distance")

    if len({turn.content for turn in scenario.conversation_turns}) < max(
        2, scenario.turn_count // 2
    ):
        report.warnings.append("more than half the turns are textually identical")

    for turn in scenario.conversation_turns:
        if not turn.expected_behavior and not turn.is_distractor:
            report.warnings.append(f"turn {turn.turn_index} has no expected_behavior for the judge")

    return report


def validate_suite(scenarios: list[Scenario], expected_count: int | None = None) -> SuiteReport:
    """Validate a whole suite and compute its composition statistics.

    Args:
        scenarios: The scenarios to check.
        expected_count: If given, a mismatch is reported as a suite error.
    """
    report = SuiteReport(total=len(scenarios))
    report.scenario_reports = [validate_scenario(scenario) for scenario in scenarios]
    report.valid = sum(1 for item in report.scenario_reports if item.valid)

    if expected_count is not None and len(scenarios) != expected_count:
        report.suite_errors.append(
            f"expected {expected_count} scenarios, generated {len(scenarios)}"
        )

    present = {scenario.category for scenario in scenarios}
    missing = REQUIRED_CATEGORIES - present
    if missing:
        report.suite_errors.append(
            f"missing required categories: {sorted(c.value for c in missing)}"
        )

    duplicates = [sid for sid, n in Counter(s.scenario_id for s in scenarios).items() if n > 1]
    if duplicates:
        report.suite_errors.append(f"duplicate scenario ids: {duplicates[:5]}")

    report.category_distribution = dict(Counter(scenario.category.value for scenario in scenarios))
    report.difficulty_distribution = dict(
        Counter(scenario.difficulty.value for scenario in scenarios)
    )
    report.length_distribution = dict(Counter(scenario.turn_count for scenario in scenarios))
    report.tool_coverage = dict(
        Counter(tool.value for scenario in scenarios for tool in scenario.expected_tool_sequence)
    )
    report.injected_failure_distribution = dict(
        Counter(
            turn.injected_failure.value
            for scenario in scenarios
            for turn in scenario.conversation_turns
            if turn.injected_failure is not InjectedFailureType.NONE
        )
    )

    covered_lengths = set(report.length_distribution)
    missing_lengths = [length for length in EXPECTED_LENGTHS if length not in covered_lengths]
    if missing_lengths:
        report.suite_warnings.append(
            f"conversation lengths not represented: {missing_lengths}; "
            "the length-sweep analysis will have gaps"
        )

    if len(report.difficulty_distribution) < 2:
        report.suite_warnings.append(
            "all scenarios share one difficulty; difficulty breakdowns will be uninformative"
        )

    for difficulty in Difficulty:
        report.difficulty_distribution.setdefault(difficulty.value, 0)

    projects = Counter(
        str(scenario.initial_state.get("project_name", "")) for scenario in scenarios
    )
    if projects and projects.most_common(1)[0][1] > len(scenarios) * 0.4:
        report.suite_warnings.append(
            "over 40 percent of scenarios use a single project name; variation is thin"
        )

    return report


def assert_valid(scenarios: list[Scenario]) -> None:
    """Raise if any scenario is unusable.

    Raises:
        ScenarioValidationError: On the first invalid scenario found.
    """
    for scenario in scenarios:
        report = validate_scenario(scenario)
        if not report.valid:
            raise ScenarioValidationError(report.scenario_id, report.errors)


def validate_suite_object(suite: ScenarioSuite) -> SuiteReport:
    """Validate a :class:`ScenarioSuite` wrapper."""
    return validate_suite(suite.scenarios)
