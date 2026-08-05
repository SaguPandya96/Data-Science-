"""Baseline-versus-candidate comparison and the regression gate.

The gate exists because "the score went down a bit" is not a decision anyone can act on.
Tolerances live in ``configs/release_thresholds.yaml`` with rationale strings, a breach
produces a named violation, and the CLI exits non-zero — so a regression stops a pipeline
rather than appearing in a report nobody reads.

Effect sizes accompany every delta. A three-point drop from 0.98 to 0.95 and from 0.50
to 0.47 are the same number and very different events; Cohen's *h* distinguishes them.
"""

from __future__ import annotations

from evalforge.analytics.metrics import RunMetrics
from evalforge.analytics.statistics import (
    cliffs_delta,
    cohens_h,
    interpret_effect,
)
from evalforge.config import EvalForgeConfig
from evalforge.exceptions import RegressionGateError
from evalforge.logging_config import get_logger
from evalforge.schemas.evaluation import ComparisonReport, MetricDelta, SessionSummary

logger = get_logger(__name__)

#: Metrics compared by default. Restricted to ones with a stated tolerance, because a
#: gate on an ungoverned metric is a gate nobody agreed to.
DEFAULT_COMPARED_METRICS: tuple[str, ...] = (
    "pass_rate",
    "overall_score",
    "task_completion",
    "context_retention",
    "instruction_adherence",
    "tool_reliability",
    "tool_selection_accuracy",
    "tool_argument_accuracy",
    "tool_sequence_accuracy",
    "recovery_success_rate",
    "prompt_injection_resistance",
    "consistency",
    "efficiency",
    "safety",
    "critical_failure_count",
    "goal_drift_rate",
    "unsupported_claim_rate",
    "contradiction_rate",
    "average_latency_ms",
    "p95_latency_ms",
    "estimated_cost_per_success",
)


def _delta(
    name: str,
    baseline: float,
    candidate: float,
    config: EvalForgeConfig,
) -> MetricDelta:
    """Build one metric delta, applying the configured tolerance and direction."""
    higher_is_better = config.release.higher_is_better(name)
    tolerance = config.release.tolerance_for(name)
    absolute = candidate - baseline
    relative = absolute / baseline if baseline else None

    # Tolerance is a signed allowance: -0.03 permits a three-point drop on a
    # higher-is-better metric, +0.05 permits a five-point rise on a lower-is-better one.
    regressed = False
    if tolerance is not None:
        regressed = absolute < tolerance if higher_is_better else absolute > tolerance

    return MetricDelta(
        name=name,
        baseline=round(baseline, 6),
        candidate=round(candidate, 6),
        absolute_change=round(absolute, 6),
        relative_change=round(relative, 6) if relative is not None else None,
        tolerance=tolerance,
        regressed=regressed,
        higher_is_better=higher_is_better,
    )


def _group_deltas(
    baseline_groups: dict[str, dict[str, float]],
    candidate_groups: dict[str, dict[str, float]],
    config: EvalForgeConfig,
    metric: str = "pass_rate",
) -> dict[str, list[MetricDelta]]:
    """Compare one metric across every subgroup present in either run."""
    output: dict[str, list[MetricDelta]] = {}
    for group in sorted(set(baseline_groups) | set(candidate_groups)):
        base = baseline_groups.get(group, {})
        cand = candidate_groups.get(group, {})
        deltas = [
            _delta(metric, base.get(metric, 0.0), cand.get(metric, 0.0), config),
            _delta(
                "overall_score",
                base.get("overall_score", 0.0),
                cand.get("overall_score", 0.0),
                config,
            ),
            _delta(
                "context_retention",
                base.get("context_retention", 0.0),
                cand.get("context_retention", 0.0),
                config,
            ),
        ]
        output[group] = deltas
    return output


def compare_runs(
    baseline_metrics: RunMetrics,
    candidate_metrics: RunMetrics,
    config: EvalForgeConfig,
    baseline_sessions: list[SessionSummary] | None = None,
    candidate_sessions: list[SessionSummary] | None = None,
    metrics: tuple[str, ...] = DEFAULT_COMPARED_METRICS,
) -> ComparisonReport:
    """Compare two evaluated runs and decide the regression gate.

    Args:
        baseline_metrics: Reference run's metrics.
        candidate_metrics: Metrics for the run under scrutiny.
        config: Supplies tolerances and metric directions.
        baseline_sessions: Optional summaries, enabling a distribution effect size.
        candidate_sessions: Optional summaries for the candidate.
        metrics: Which metric names to compare.

    Returns:
        A :class:`ComparisonReport` whose ``gate_passed`` is false if any tolerance
        was breached.
    """
    deltas = [
        _delta(name, baseline_metrics.get(name), candidate_metrics.get(name), config)
        for name in metrics
    ]

    regressions: list[str] = []
    improvements: list[str] = []

    for delta in deltas:
        if delta.regressed:
            direction = "fell" if delta.higher_is_better else "rose"
            effect = ""
            if 0.0 <= delta.baseline <= 1.0 and 0.0 <= delta.candidate <= 1.0:
                h = cohens_h(delta.candidate, delta.baseline)
                effect = f", effect size h={h:+.3f} ({interpret_effect(h)})"
            regressions.append(
                f"{delta.name}: {delta.baseline:.4f} -> {delta.candidate:.4f} "
                f"({direction} {abs(delta.absolute_change):.4f}, "
                f"allowed {delta.tolerance}){effect}"
            )
        elif delta.higher_is_better and delta.absolute_change > 0.01:
            improvements.append(
                f"{delta.name}: {delta.baseline:.4f} -> {delta.candidate:.4f} "
                f"(+{delta.absolute_change:.4f})"
            )

    report = ComparisonReport(
        baseline_run_id=baseline_metrics.run_id,
        candidate_run_id=candidate_metrics.run_id,
        deltas=deltas,
        deltas_by_category=_group_deltas(
            baseline_metrics.by_category, candidate_metrics.by_category, config
        ),
        deltas_by_difficulty=_group_deltas(
            baseline_metrics.by_difficulty, candidate_metrics.by_difficulty, config
        ),
        deltas_by_length=_group_deltas(
            baseline_metrics.by_length, candidate_metrics.by_length, config
        ),
        regressions=regressions,
        improvements=improvements,
        gate_passed=not regressions,
    )

    if baseline_sessions and candidate_sessions:
        delta_value = cliffs_delta(
            [item.overall_score for item in candidate_sessions],
            [item.overall_score for item in baseline_sessions],
        )
        report.metadata["cliffs_delta_overall_score"] = round(delta_value, 4)
        report.metadata["cliffs_delta_interpretation"] = interpret_effect(delta_value)
        report.metadata["baseline_sessions"] = len(baseline_sessions)
        report.metadata["candidate_sessions"] = len(candidate_sessions)

    report.metadata["compared_metrics"] = len(deltas)
    report.metadata["gated_metrics"] = sum(1 for d in deltas if d.tolerance is not None)

    logger.info(
        "comparison_complete",
        baseline=baseline_metrics.run_id,
        candidate=candidate_metrics.run_id,
        regressions=len(regressions),
        gate_passed=report.gate_passed,
    )
    return report


def enforce_gate(report: ComparisonReport) -> None:
    """Raise if the candidate regressed beyond tolerance.

    Raises:
        RegressionGateError: With one message per breached metric.
    """
    if not report.gate_passed:
        raise RegressionGateError(report.regressions)


def render_comparison_table(report: ComparisonReport) -> list[tuple[str, str, str, str, str]]:
    """Rows for a terminal or dashboard comparison table.

    Returns:
        ``(metric, baseline, candidate, change, verdict)`` tuples.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for delta in report.deltas:
        if delta.regressed:
            verdict = "REGRESSED"
        elif delta.tolerance is None:
            verdict = "not gated"
        else:
            verdict = "ok"
        rows.append(
            (
                delta.name,
                f"{delta.baseline:.4f}",
                f"{delta.candidate:.4f}",
                f"{delta.absolute_change:+.4f}",
                verdict,
            )
        )
    return rows
