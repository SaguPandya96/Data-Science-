"""Metric computation over an evaluated run.

Everything here is derived from stored ``SessionSummary`` and ``EvaluationResult``
records, never from live objects, so the same numbers can be recomputed months later
from the database alone.

Metrics are grouped by *what they answer*, not by evaluator, because that is how the
release report and dashboard consume them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from evalforge.analytics.statistics import (
    Interval,
    bootstrap_mean,
    percentile,
    rate,
    summarize_distribution,
    wilson_interval,
)
from evalforge.schemas.common import Dimension, FailureCategory
from evalforge.schemas.evaluation import EvaluationResult, SessionSummary

#: Conversation-length buckets used by the length-sweep analysis.
LENGTH_BUCKETS: tuple[int, ...] = (5, 10, 15, 20, 30)


def bucket_length(turn_count: int) -> int:
    """Snap a turn count to its nearest analysis bucket."""
    return min(LENGTH_BUCKETS, key=lambda bucket: abs(bucket - turn_count))


@dataclass
class RunMetrics:
    """Every metric computed for one run.

    Args:
        run_id: Run these metrics describe.
        session_count: Sessions evaluated.
        scalars: Flat metric name to value, the form the release gate consumes.
        intervals: Uncertainty intervals for metrics that have one.
        by_category: Metrics grouped by scenario category.
        by_difficulty: Metrics grouped by difficulty.
        by_length: Metrics grouped by conversation-length bucket.
        failure_counts: Occurrences of each failure category.
        tool_reliability: Tool-specific accuracy metrics.
        latency: Latency distribution summary.
    """

    run_id: str
    session_count: int = 0
    scalars: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, Interval] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_length: dict[str, dict[str, float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    by_prompt_version: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    tool_reliability: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    cascade: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        """Look up one scalar metric."""
        return self.scalars.get(name, default)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for JSON reports."""
        return {
            "run_id": self.run_id,
            "session_count": self.session_count,
            "scalars": self.scalars,
            "intervals": {name: item.as_dict() for name, item in self.intervals.items()},
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "by_length": self.by_length,
            "by_model": self.by_model,
            "by_prompt_version": self.by_prompt_version,
            "failure_counts": self.failure_counts,
            "tool_reliability": self.tool_reliability,
            "latency": self.latency,
            "cascade": self.cascade,
        }


def _group_scores(summaries: list[SessionSummary], key: Any) -> dict[str, dict[str, float]]:
    """Compute a standard metric block for each group of sessions."""
    grouped: dict[str, list[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[str(key(summary))].append(summary)

    output: dict[str, dict[str, float]] = {}
    for name, group in sorted(grouped.items()):
        passed = sum(1 for item in group if item.passed)
        output[name] = {
            # n is always reported alongside: a 100% pass rate over 3 sessions and over
            # 40 sessions are very different claims, and a table without n hides that.
            "n": float(len(group)),
            "pass_rate": rate(passed, len(group)),
            "overall_score": _mean(item.overall_score for item in group),
            "context_retention": _mean(item.context_retention_score for item in group),
            "instruction_adherence": _mean(item.instruction_adherence_score for item in group),
            "tool_reliability": _mean(item.tool_reliability_score for item in group),
            "task_completion": _mean(item.task_completion_score for item in group),
            "recovery": _mean(item.recovery_score for item in group),
            "safety": _mean(item.safety_score for item in group),
            "critical_failures": float(sum(len(item.critical_failures) for item in group)),
            "mean_latency_ms": _mean(item.total_latency for item in group),
        }
    return output


def _mean(values: Any) -> float:
    """Mean of an iterable, 0.0 when empty."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _evaluator_rate(results: list[EvaluationResult], evaluator: str) -> float:
    """Mean roll-up score for one evaluator, one observation per session.

    Only roll-up results count. An evaluator also emits one result per individual
    failure, and those are session-scoped; averaging them in would weight a session by
    how many ways it failed rather than treating each session as one observation.
    """
    scores = [
        result.score
        for result in results
        if result.evaluator_name == evaluator and result.metadata.get("rollup") is True
    ]
    return _mean(scores)


def _failure_rate(summaries: list[SessionSummary], category: FailureCategory) -> float:
    """Fraction of sessions exhibiting a failure category."""
    if not summaries:
        return 0.0
    affected = sum(1 for item in summaries if category.value in item.failure_categories)
    return affected / len(summaries)


def compute_metrics(
    run_id: str,
    summaries: list[SessionSummary],
    results: list[EvaluationResult],
    seed: int = 42,
) -> RunMetrics:
    """Compute every metric for one run.

    Args:
        run_id: Run being summarised.
        summaries: One summary per evaluated session.
        results: Every evaluation result in the run.
        seed: Seed for bootstrap resampling, so intervals are reproducible.
    """
    metrics = RunMetrics(run_id=run_id, session_count=len(summaries))
    if not summaries:
        return metrics

    deterministic = [r for r in results if r.evaluator_kind == "deterministic"]
    passed = sum(1 for item in summaries if item.passed)
    scores = [item.overall_score for item in summaries]
    latencies = [item.total_latency for item in summaries]

    critical_total = sum(len(item.critical_failures) for item in summaries)
    sessions_with_critical = sum(1 for item in summaries if item.critical_failures)

    # --- headline rates -----------------------------------------------------
    metrics.scalars["pass_rate"] = rate(passed, len(summaries))
    metrics.scalars["overall_score"] = _mean(scores)
    metrics.scalars["task_completion"] = _mean(i.task_completion_score for i in summaries)
    metrics.scalars["context_retention"] = _mean(i.context_retention_score for i in summaries)
    metrics.scalars["instruction_adherence"] = _mean(
        i.instruction_adherence_score for i in summaries
    )
    metrics.scalars["tool_reliability"] = _mean(i.tool_reliability_score for i in summaries)
    metrics.scalars["recovery_success"] = _mean(i.recovery_score for i in summaries)
    metrics.scalars["consistency"] = _mean(i.consistency_score for i in summaries)
    metrics.scalars["efficiency"] = _mean(i.efficiency_score for i in summaries)
    metrics.scalars["safety"] = _mean(i.safety_score for i in summaries)

    # --- tool-use detail ----------------------------------------------------
    metrics.scalars["tool_selection_accuracy"] = _evaluator_rate(deterministic, "tool_selection")
    metrics.scalars["tool_argument_accuracy"] = _evaluator_rate(deterministic, "tool_argument")
    metrics.scalars["tool_sequence_accuracy"] = _evaluator_rate(deterministic, "tool_sequence")
    metrics.scalars["recovery_success_rate"] = _evaluator_rate(deterministic, "recovery")
    metrics.tool_reliability = {
        "selection_accuracy": metrics.scalars["tool_selection_accuracy"],
        "argument_accuracy": metrics.scalars["tool_argument_accuracy"],
        "sequence_accuracy": metrics.scalars["tool_sequence_accuracy"],
        "duplicate_free_rate": _evaluator_rate(deterministic, "duplicate_tool_call"),
        "required_step_completion": _evaluator_rate(deterministic, "required_step"),
        "retry_discipline": _evaluator_rate(deterministic, "retry_limit"),
    }

    # --- integrity and safety ----------------------------------------------
    metrics.scalars["unsupported_claim_rate"] = _failure_rate(
        summaries, FailureCategory.UNSUPPORTED_CLAIM
    )
    metrics.scalars["goal_drift_rate"] = _failure_rate(summaries, FailureCategory.GOAL_DRIFT)
    metrics.scalars["contradiction_rate"] = _failure_rate(
        summaries, FailureCategory.INTERNAL_CONTRADICTION
    )
    metrics.scalars["unnecessary_tool_call_rate"] = _failure_rate(
        summaries, FailureCategory.UNNECESSARY_TOOL_CALL
    )
    injection_sessions = [
        item for item in summaries if item.scenario_category == "prompt_injection"
    ]
    injection_failures = sum(
        1
        for item in summaries
        if FailureCategory.PROMPT_INJECTION_FOLLOWED.value in item.failure_categories
    )
    metrics.scalars["prompt_injection_failure_rate"] = rate(injection_failures, len(summaries))
    # Resistance is measured against every session, not only injection-category ones:
    # an agent that obeys an injection in a stress scenario has the same defect.
    metrics.scalars["prompt_injection_resistance"] = (
        1.0 - metrics.scalars["prompt_injection_failure_rate"]
    )
    metrics.scalars["injection_scenarios"] = float(len(injection_sessions))

    # --- critical failures --------------------------------------------------
    metrics.scalars["critical_failures"] = float(critical_total)
    metrics.scalars["critical_failure_count"] = float(critical_total)
    metrics.scalars["sessions_with_critical_failure"] = float(sessions_with_critical)
    metrics.scalars["critical_failure_session_rate"] = rate(sessions_with_critical, len(summaries))

    # --- cost and latency ---------------------------------------------------
    metrics.latency = summarize_distribution(latencies)
    metrics.scalars["average_latency_ms"] = metrics.latency["mean"]
    metrics.scalars["p50_latency_ms"] = percentile(latencies, 50)
    metrics.scalars["p95_latency_ms"] = percentile(latencies, 95)
    metrics.scalars["total_tokens"] = float(sum(item.total_tokens for item in summaries))
    metrics.scalars["estimated_cost"] = sum(item.estimated_cost for item in summaries)
    metrics.scalars["estimated_cost_per_success"] = (
        metrics.scalars["estimated_cost"] / passed if passed else 0.0
    )

    # --- cascading errors ---------------------------------------------------
    cascade_results = [r for r in deterministic if r.evaluator_name == "cascading_error"]
    depths = [
        float(r.metadata.get("propagation_depth", 0))
        for r in cascade_results
        if r.evaluation_level.value == "session"
    ]
    impacts = [
        float(r.metadata.get("downstream_impact", 0.0))
        for r in cascade_results
        if r.evaluation_level.value == "session"
    ]
    metrics.cascade = {
        "sessions_with_corruption": float(len(depths)),
        "mean_propagation_depth": _mean(depths),
        "max_propagation_depth": max(depths) if depths else 0.0,
        "mean_downstream_impact": _mean(impacts),
    }
    metrics.scalars["error_propagation_depth"] = metrics.cascade["mean_propagation_depth"]
    metrics.scalars["downstream_impact_score"] = metrics.cascade["mean_downstream_impact"]

    # --- uncertainty --------------------------------------------------------
    metrics.intervals["pass_rate"] = wilson_interval(passed, len(summaries))
    metrics.intervals["overall_score"] = bootstrap_mean(scores, seed=seed)
    metrics.intervals["context_retention"] = bootstrap_mean(
        [i.context_retention_score for i in summaries], seed=seed
    )
    metrics.intervals["instruction_adherence"] = bootstrap_mean(
        [i.instruction_adherence_score for i in summaries], seed=seed
    )
    metrics.intervals["prompt_injection_resistance"] = wilson_interval(
        len(summaries) - injection_failures, len(summaries)
    )

    # --- breakdowns ---------------------------------------------------------
    metrics.by_category = _group_scores(summaries, lambda s: s.scenario_category)
    metrics.by_difficulty = _group_scores(summaries, lambda s: s.scenario_difficulty)
    metrics.by_length = _group_scores(summaries, lambda s: bucket_length(s.turn_count))
    metrics.by_model = _group_scores(summaries, lambda s: s.model_name)
    metrics.by_prompt_version = _group_scores(summaries, lambda s: s.prompt_version)

    counter: Counter[str] = Counter()
    for summary in summaries:
        counter.update(summary.failure_categories)
    metrics.failure_counts = dict(counter.most_common())

    return metrics


def dimension_breakdown(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    """Per-dimension mean score and failure count, for the dashboard."""
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        if result.evaluator_kind == "deterministic":
            grouped[result.dimension.value].append(result)

    output: dict[str, dict[str, float]] = {}
    for name, group in sorted(grouped.items()):
        output[name] = {
            "mean_score": _mean(item.score for item in group),
            "checks": float(len(group)),
            "failures": float(sum(1 for item in group if not item.passed)),
        }
    return output


def failure_examples(
    results: list[EvaluationResult], limit_per_category: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Representative failing results per category, for the dashboard and report.

    The most severe examples are kept, since those are the ones a reader needs to see
    before deciding whether a category matters.
    """
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    severity_order = {"critical": 0, "major": 1, "minor": 2, "info": 3}

    for result in results:
        if result.passed or result.failure_category is FailureCategory.NONE:
            continue
        grouped[result.failure_category.value].append(result)

    output: dict[str, list[dict[str, Any]]] = {}
    for category, group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda r: (severity_order.get(r.severity.value, 9), r.score))
        output[category] = [
            {
                "scenario_id": item.scenario_id,
                "session_id": item.session_id,
                "turn_index": item.turn_index,
                "evaluator": item.evaluator_name,
                "severity": item.severity.value,
                "score": item.score,
                "reasoning": item.reasoning_summary,
                "evidence": [
                    {"kind": e.kind, "excerpt": e.excerpt, "detail": e.detail}
                    for e in item.evidence[:2]
                ],
            }
            for item in ranked[:limit_per_category]
        ]
    return output


def judge_agreement_with_deterministic(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """How often the judge and the deterministic checks reach the same verdict.

    Reported per dimension, and never used to adjust either score. It is a diagnostic on
    the *judge*: a dimension where the judge routinely disagrees with an exact check is
    a dimension where the judge should not be trusted on questions that have no exact
    check either.
    """
    deterministic: dict[tuple[str, str], list[bool]] = defaultdict(list)
    judged: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for result in results:
        if result.evaluation_level.value != "session":
            continue
        key = (result.session_id, result.dimension.value)
        if result.evaluator_kind == "deterministic":
            deterministic[key].append(result.passed)
        elif result.evaluator_kind == "judge":
            judged[key].append(result.passed)

    per_dimension: dict[str, list[bool]] = defaultdict(list)
    for key, judge_verdicts in judged.items():
        if key not in deterministic:
            continue
        auto_pass = all(deterministic[key])
        judge_pass = all(judge_verdicts)
        per_dimension[key[1]].append(auto_pass == judge_pass)

    return {
        dimension: _mean(1.0 if agreed else 0.0 for agreed in agreements)
        for dimension, agreements in sorted(per_dimension.items())
    }


def scored_dimension_names() -> list[str]:
    """Names of the dimensions that contribute to the overall score."""
    return [
        Dimension.TASK_COMPLETION.value,
        Dimension.CONTEXT_RETENTION.value,
        Dimension.INSTRUCTION_ADHERENCE.value,
        Dimension.TOOL_RELIABILITY.value,
        Dimension.RECOVERY_QUALITY.value,
        Dimension.CONSISTENCY.value,
        Dimension.EFFICIENCY.value,
        Dimension.SAFETY.value,
    ]
