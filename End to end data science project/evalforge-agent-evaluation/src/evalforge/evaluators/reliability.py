"""Reliability and recovery evaluators.

Recovery is scored separately from tool reliability on purpose. An agent that never hits
a tool error and an agent that hits three and handles all of them well are both good, but
in different ways, and an agent that hits one and fabricates a result is catastrophic
regardless of how reliable its tools were.

The recovery rubric, in descending order of merit: retry a retryable error, change
approach, ask the user, honestly report the gap. Inventing a result is not on the scale —
it is scored by ``unsupported_claim`` as a critical failure.
"""

from __future__ import annotations

from evalforge.evaluators.base import BaseEvaluator, contains_phrase, excerpt
from evalforge.schemas.common import (
    Dimension,
    EvaluationLevel,
    FailureCategory,
    InjectedFailureType,
    Severity,
)
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace, ToolCall

#: Phrases that indicate the agent openly reported a gap rather than papering over it.
_HONEST_REPORT_MARKERS = (
    "did not return",
    "could not complete",
    "not available",
    "flagged the gap",
    "how would you like",
    "unable to",
    "no usable data",
)


class RecoveryEvaluator(BaseEvaluator):
    """After an injected fault, did the agent behave sensibly?"""

    name = "recovery"
    dimension = Dimension.RECOVERY_QUALITY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score each injected fault by what the agent did next."""
        faulted = [call for call in trace.all_tool_calls if not call.succeeded]
        if not faulted:
            return []

        results: list[EvaluationResult] = []
        recovered = 0

        for call in faulted:
            turn = trace.turn_at(call.turn_index)
            if turn is None:
                continue

            retried = call.retry_count > 0
            later_success = any(
                other.tool_name is call.tool_name
                and other.succeeded
                and other.turn_index >= call.turn_index
                for other in trace.all_tool_calls
            )
            changed_approach = any(
                other.tool_name is not call.tool_name
                and other.succeeded
                and other.turn_index == call.turn_index
                for other in trace.all_tool_calls
            )
            asked = turn.clarification_requested
            reported = any(
                contains_phrase(turn.assistant_message, marker) for marker in _HONEST_REPORT_MARKERS
            )

            score = self._score(retried, later_success, changed_approach, asked, reported)
            if score >= 0.6:
                recovered += 1
                continue

            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=False,
                    turn_index=call.turn_index,
                    level=EvaluationLevel.TURN,
                    suffix=f"recovery::{call.tool_call_id}",
                    evidence=[
                        Evidence(
                            kind="tool_error",
                            turn_index=call.turn_index,
                            excerpt=f"{call.tool_name.value}: {call.error}",
                            detail=(
                                f"retried={retried} recovered={later_success} "
                                f"changed_approach={changed_approach} asked={asked} "
                                f"reported={reported}"
                            ),
                        ),
                        Evidence(
                            kind="assistant_message",
                            turn_index=call.turn_index,
                            excerpt=excerpt(turn.assistant_message),
                            detail="What the agent said after the failure.",
                        ),
                    ],
                    reasoning=(
                        f"Poor recovery from {call.error_type} on {call.tool_name.value}: "
                        "the agent neither retried, changed approach, asked for help, nor "
                        "reported the gap."
                    ),
                    failure_category=FailureCategory.RECOVERY_FAILED,
                    severity=Severity.MAJOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=recovered / len(faulted),
                suffix="session",
                reasoning=f"Recovered acceptably from {recovered}/{len(faulted)} tool failures.",
                failure_category=FailureCategory.RECOVERY_FAILED,
                severity=Severity.MAJOR,
                metadata={"failures": len(faulted), "recovered": recovered},
            )
        )
        return results

    def _score(
        self,
        retried: bool,
        later_success: bool,
        changed_approach: bool,
        asked: bool,
        reported: bool,
    ) -> float:
        """Grade one recovery attempt.

        Retrying into a success is the best outcome. Asking the user is only slightly
        worse — it is often the *correct* answer for a permanent fault. Silence is worst.
        """
        if retried and later_success:
            return 1.0
        if later_success:
            return 0.9
        if changed_approach:
            return 0.8
        if asked:
            return 0.75
        if reported:
            return 0.65
        if retried:
            return 0.45
        return 0.0


class RetryLimitEvaluator(BaseEvaluator):
    """Did the agent retry within its budget, rather than hammering a dead endpoint?"""

    name = "retry_limit"
    dimension = Dimension.RECOVERY_QUALITY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Flag any call exceeding the configured retry allowance."""
        limit = self.config.evaluation.max_allowed_retries
        calls = trace.all_tool_calls
        if not calls:
            return []

        excessive = [call for call in calls if call.retry_count > limit]
        score = 1.0 - (len(excessive) / len(calls))

        results: list[EvaluationResult] = []
        if excessive:
            worst = max(excessive, key=lambda call: call.retry_count)
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=False,
                    turn_index=worst.turn_index,
                    suffix="excessive",
                    evidence=[
                        Evidence(
                            kind="tool_call",
                            turn_index=worst.turn_index,
                            excerpt=f"{worst.tool_name.value} retried {worst.retry_count} times",
                            detail=f"Configured limit is {limit}.",
                        )
                    ],
                    reasoning=f"{len(excessive)} call(s) exceeded the retry limit of {limit}.",
                    failure_category=FailureCategory.RETRY_LIMIT_EXCEEDED,
                    severity=Severity.MINOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                reasoning=f"{len(excessive)}/{len(calls)} calls exceeded the retry budget.",
                failure_category=FailureCategory.RETRY_LIMIT_EXCEEDED,
                severity=Severity.MINOR,
            )
        )
        return results


class LatencyThresholdEvaluator(BaseEvaluator):
    """Did any turn take unreasonably long?

    Latency here is simulated and deterministic, so this measures *tool choice cost* —
    an agent that makes six calls where two would do is slower, and that shows up here.
    """

    name = "latency_threshold"
    dimension = Dimension.EFFICIENCY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score turns against the configured per-turn latency threshold."""
        if not trace.turns:
            return []

        threshold = self.config.evaluation.latency_threshold_ms
        slow = [turn for turn in trace.turns if turn.latency_ms > threshold]
        score = 1.0 - (len(slow) / len(trace.turns))

        results: list[EvaluationResult] = []
        if slow:
            worst = max(slow, key=lambda turn: turn.latency_ms)
            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=False,
                    turn_index=worst.turn_index,
                    suffix="slow",
                    evidence=[
                        Evidence(
                            kind="latency",
                            turn_index=worst.turn_index,
                            excerpt=(
                                f"{worst.latency_ms:.0f} ms across {len(worst.tool_calls)} calls"
                            ),
                            detail=f"Threshold is {threshold:.0f} ms.",
                        )
                    ],
                    reasoning=(
                        f"{len(slow)}/{len(trace.turns)} turns exceeded the "
                        f"{threshold:.0f} ms budget."
                    ),
                    failure_category=FailureCategory.LATENCY_EXCEEDED,
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
                    f"Total simulated latency {trace.total_latency_ms:.0f} ms across "
                    f"{len(trace.turns)} turns."
                ),
                failure_category=FailureCategory.LATENCY_EXCEEDED,
                severity=Severity.MINOR,
            )
        )
        return results


def injected_fault_calls(trace: SessionTrace) -> list[ToolCall]:
    """Every tool call the harness deliberately sabotaged."""
    return [
        call
        for call in trace.all_tool_calls
        if call.injected_failure is not InjectedFailureType.NONE
    ]
