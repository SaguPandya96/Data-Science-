"""Tool-use evaluators: selection, arguments, order, redundancy and step completion.

The five checks here deliberately measure different things. An agent can pick the right
tools in the wrong order, or the right tools in the right order with a wrong argument, or
do everything correctly twice. Collapsing these into one "tool accuracy" number would
hide which of them is happening, and they have different fixes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalforge.evaluators.base import (
    BaseEvaluator,
    dates_match,
    lcs_ratio,
    normalize_money,
    normalize_text,
)
from evalforge.schemas.common import Dimension, FailureCategory, Severity, ToolName
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace, ToolCall


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compare only the argument keys the scenario pinned.

    Pinning every field would measure prompt-template stability rather than correctness,
    so a scenario declares the values it cares about and the rest are free.

    Returns:
        Whether every pinned key matched, and the list of keys that did not.
    """
    mismatched: list[str] = []
    for key, want in expected.items():
        got = actual.get(key)
        if got is None:
            mismatched.append(key)
            continue
        if isinstance(want, int | float) and not isinstance(want, bool):
            got_value = normalize_money(got)
            if got_value is None or abs(got_value - float(want)) > 0.01:
                mismatched.append(key)
            continue
        if isinstance(want, str) and _looks_like_date(want):
            if not dates_match(want, str(got)):
                mismatched.append(key)
            continue
        if normalize_text(str(want)) not in normalize_text(str(got)):
            mismatched.append(key)
    return not mismatched, mismatched


def _looks_like_date(value: str) -> bool:
    """Cheap check for a date-shaped string."""
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _expected_calls(scenario: Scenario) -> list[tuple[int, Any]]:
    """Every expected tool call with the turn index it belongs to."""
    return [
        (turn.turn_index, expected)
        for turn in scenario.conversation_turns
        for expected in turn.expected_tool_calls
    ]


class ToolSelectionEvaluator(BaseEvaluator):
    """Did the agent call the tools the contract required, and no phantom ones?"""

    name = "tool_selection"
    dimension = Dimension.TOOL_RELIABILITY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score required-call coverage, penalised by calls with no contract entry."""
        expected = [item for item in _expected_calls(scenario) if item[1].required]
        if not expected:
            return []

        actual = trace.all_tool_calls
        actual_by_turn: dict[int, list[ToolCall]] = {}
        for call in actual:
            actual_by_turn.setdefault(call.turn_index, []).append(call)

        satisfied = 0
        missing: list[str] = []
        for turn_index, want in expected:
            # A tool may legitimately land a turn later than scripted when the agent
            # first has to recover from a failure, so a small window is allowed.
            window = [
                call
                for index in range(turn_index, turn_index + 2)
                for call in actual_by_turn.get(index, [])
            ]
            if any(call.tool_name is want.tool_name for call in window):
                satisfied += 1
            else:
                missing.append(f"{want.tool_name.value}@turn{turn_index}")

        contract_tools = {want.tool_name for _, want in _expected_calls(scenario)}
        unexpected = [call for call in actual if call.tool_name not in contract_tools]

        coverage = satisfied / len(expected)
        penalty = min(0.3, 0.1 * len(unexpected))
        score = max(0.0, coverage - penalty)

        results: list[EvaluationResult] = []
        if missing:
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=0.0,
                    passed=False,
                    suffix="missing",
                    evidence=[
                        Evidence(
                            kind="tool_sequence",
                            excerpt=str([c.value for c in trace.tool_sequence]),
                            detail=f"Missing required calls: {missing}",
                        )
                    ],
                    reasoning=f"{len(missing)} required tool call(s) never happened: {missing}.",
                    failure_category=FailureCategory.MISSING_TOOL_CALL,
                    severity=Severity.MAJOR,
                )
            )
        if unexpected:
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=max(0.0, 1.0 - 0.2 * len(unexpected)),
                    passed=False,
                    suffix="unexpected",
                    evidence=[
                        Evidence(
                            kind="tool_call",
                            turn_index=unexpected[0].turn_index,
                            excerpt=f"{unexpected[0].tool_name.value}({unexpected[0].arguments})",
                            detail="No contract entry for this tool in this scenario.",
                        )
                    ],
                    reasoning=(
                        f"{len(unexpected)} call(s) to tools the scenario never asked for: "
                        f"{sorted({c.tool_name.value for c in unexpected})}."
                    ),
                    failure_category=FailureCategory.WRONG_TOOL_SELECTED,
                    severity=Severity.MAJOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                reasoning=(
                    f"Covered {satisfied}/{len(expected)} required calls; "
                    f"{len(unexpected)} unexpected call(s)."
                ),
                failure_category=FailureCategory.WRONG_TOOL_SELECTED,
                severity=Severity.MAJOR,
                metadata={"coverage": coverage, "unexpected": len(unexpected)},
            )
        )
        return results


class ToolArgumentEvaluator(BaseEvaluator):
    """Did the agent pass the right values to the right tools?"""

    name = "tool_argument"
    dimension = Dimension.TOOL_RELIABILITY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Exact-match the pinned argument subset for every expected call."""
        expected = [item for item in _expected_calls(scenario) if item[1].arguments]
        if not expected:
            return []

        results: list[EvaluationResult] = []
        matched = 0
        checked = 0

        for turn_index, want in expected:
            candidates = [
                call
                for call in trace.all_tool_calls
                if call.tool_name is want.tool_name
                and turn_index <= call.turn_index <= turn_index + 1
            ]
            if not candidates:
                continue
            checked += 1
            call = candidates[0]
            ok, mismatched = arguments_match(want.arguments, call.arguments)
            if ok:
                matched += 1
                continue

            entity_error = any(key in {"project_name", "plan"} for key in mismatched)
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=0.0,
                    passed=False,
                    turn_index=call.turn_index,
                    suffix=f"args::{call.tool_call_id}",
                    evidence=[
                        Evidence(
                            kind="tool_call",
                            turn_index=call.turn_index,
                            excerpt=(
                                f"{call.tool_name.value}"
                                f"({ {k: call.arguments.get(k) for k in mismatched} })"
                            ),
                            detail=f"Expected {[(k, want.arguments[k]) for k in mismatched]}",
                        )
                    ],
                    reasoning=(
                        f"{call.tool_name.value} received wrong value(s) for "
                        f"{', '.join(mismatched)} at turn {call.turn_index}."
                    ),
                    failure_category=(
                        FailureCategory.WRONG_ENTITY_SELECTED
                        if entity_error
                        else FailureCategory.WRONG_TOOL_ARGUMENT
                    ),
                    severity=Severity.MAJOR,
                )
            )

        if not checked:
            return results

        results.append(
            self.make_result(
                scenario,
                trace,
                score=matched / checked,
                suffix="session",
                reasoning=f"{matched}/{checked} tool calls had exactly correct pinned arguments.",
                failure_category=FailureCategory.WRONG_TOOL_ARGUMENT,
                severity=Severity.MAJOR,
            )
        )
        return results


class ToolSequenceEvaluator(BaseEvaluator):
    """Did the agent respect the required ordering of dependent steps?"""

    name = "tool_sequence"
    dimension = Dimension.TOOL_RELIABILITY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score order as an LCS ratio against the expected sequence."""
        expected = [tool.value for tool in scenario.expected_tool_sequence]
        if not expected:
            return []

        actual = [tool.value for tool in trace.tool_sequence]
        score = lcs_ratio(expected, actual)

        return [
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                evidence=[
                    Evidence(
                        kind="tool_sequence",
                        excerpt=f"expected={expected[:12]} actual={actual[:12]}",
                        detail="Scored as longest-common-subsequence overlap, not exact equality.",
                    )
                ],
                reasoning=(
                    f"Tool order matched the expected sequence at {score:.0%} "
                    f"({len(actual)} calls made, {len(expected)} expected)."
                ),
                failure_category=FailureCategory.WRONG_TOOL_SEQUENCE,
                severity=Severity.MINOR,
            )
        ]


class DuplicateToolCallEvaluator(BaseEvaluator):
    """Did the agent repeat itself, or call tools that changed nothing?"""

    name = "duplicate_tool_call"
    dimension = Dimension.EFFICIENCY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Count identical repeat calls and no-op calls, excluding legitimate retries."""
        calls = trace.all_tool_calls
        if not calls:
            return []

        signatures = Counter(
            (call.turn_index, call.tool_name.value, _signature(call.arguments))
            for call in calls
            # A retry is a deliberate recovery action, not redundancy. Counting it here
            # would penalise exactly the behaviour the recovery evaluator rewards.
            if call.retry_count == 0
        )
        duplicates = {key: count for key, count in signatures.items() if count > 1}
        duplicate_calls = sum(count - 1 for count in duplicates.values())

        expected_tools = set(scenario.expected_tool_sequence)
        unnecessary = [
            call
            for call in calls
            if call.tool_name not in expected_tools and call.succeeded and not _changed_state(call)
        ]

        total = len(calls)
        waste = duplicate_calls + len(unnecessary)
        score = max(0.0, 1.0 - waste / total)

        results: list[EvaluationResult] = []
        if duplicate_calls:
            first_key = next(iter(duplicates))
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=False,
                    turn_index=int(first_key[0]),
                    suffix="duplicates",
                    evidence=[
                        Evidence(
                            kind="tool_call",
                            turn_index=int(first_key[0]),
                            excerpt=(
                                f"{first_key[1]} called {duplicates[first_key]} times identically"
                            ),
                            detail="Same tool, same arguments, same turn, not a retry.",
                        )
                    ],
                    reasoning=f"{duplicate_calls} redundant duplicate tool call(s).",
                    failure_category=FailureCategory.DUPLICATE_TOOL_CALL,
                    severity=Severity.MINOR,
                )
            )
        if unnecessary:
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=False,
                    turn_index=unnecessary[0].turn_index,
                    suffix="unnecessary",
                    reasoning=(
                        f"{len(unnecessary)} tool call(s) outside the contract that changed "
                        "no state."
                    ),
                    failure_category=FailureCategory.UNNECESSARY_TOOL_CALL,
                    severity=Severity.MINOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                reasoning=(
                    f"{waste} wasted call(s) out of {total} "
                    f"({duplicate_calls} duplicate, {len(unnecessary)} unnecessary)."
                ),
                failure_category=FailureCategory.DUPLICATE_TOOL_CALL,
                severity=Severity.MINOR,
                metadata={"duplicates": duplicate_calls, "unnecessary": len(unnecessary)},
            )
        )
        return results


class RequiredStepEvaluator(BaseEvaluator):
    """Did the session actually finish the job?

    Distinct from tool selection: selection asks whether individual calls happened,
    this asks whether the *workflow* completed and produced its deliverable.
    """

    name = "required_step"
    dimension = Dimension.TASK_COMPLETION

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score exercised required tools plus final-artifact production."""
        required = set(scenario.required_tools)
        if not required:
            return []

        exercised = {call.tool_name for call in trace.all_tool_calls if call.succeeded}
        covered = required & exercised
        missing = sorted(tool.value for tool in required - exercised)

        wanted_summary = ToolName.DRAFT_EXECUTIVE_SUMMARY in required
        produced_summary = any(
            call.tool_name is ToolName.DRAFT_EXECUTIVE_SUMMARY and call.succeeded
            for call in trace.all_tool_calls
        )
        has_artifact = trace.final_artifact is not None

        step_score = len(covered) / len(required)
        artifact_score = 1.0 if (has_artifact and (produced_summary or not wanted_summary)) else 0.0
        score = 0.7 * step_score + 0.3 * artifact_score

        results: list[EvaluationResult] = []
        if missing:
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=step_score,
                    passed=False,
                    suffix="incomplete",
                    evidence=[
                        Evidence(
                            kind="tool_sequence",
                            excerpt=str(sorted(tool.value for tool in exercised)),
                            detail=f"Required but never completed: {missing}",
                        )
                    ],
                    reasoning=(
                        f"Workflow incomplete; never successfully used: {', '.join(missing)}."
                    ),
                    failure_category=FailureCategory.OBJECTIVE_INCOMPLETE,
                    severity=Severity.MAJOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                reasoning=(
                    f"Completed {len(covered)}/{len(required)} required steps; "
                    f"final artifact {'produced' if has_artifact else 'missing'}."
                ),
                failure_category=FailureCategory.OBJECTIVE_INCOMPLETE,
                severity=Severity.MAJOR,
            )
        )
        return results


def _signature(arguments: dict[str, Any]) -> str:
    """Stable, comparable rendering of a call's arguments."""
    return repr(sorted((key, str(value)[:80]) for key, value in arguments.items()))


def _changed_state(call: ToolCall) -> bool:
    """Whether a call plausibly altered agent state."""
    if call.tool_name in {
        ToolName.CREATE_PROJECT_PLAN,
        ToolName.UPDATE_PROJECT_PLAN,
        ToolName.SAVE_ARTIFACT,
        ToolName.DRAFT_EXECUTIVE_SUMMARY,
        ToolName.DRAFT_STAKEHOLDER_EMAIL,
    }:
        return True
    result = call.result or {}
    return bool(result.get("documents") or result.get("requirements"))
