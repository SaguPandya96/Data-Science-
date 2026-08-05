"""Evaluation schema: results, session summaries and run summaries.

An ``EvaluationResult`` is deliberately evidence-bearing. A score with no pointer back
into the trace is unauditable, so every result carries the spans that justified it —
which is also what makes human/automated disagreement analysis possible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalforge.schemas.common import (
    Dimension,
    EvaluationLevel,
    FailureCategory,
    ReleaseDecision,
    Severity,
    is_critical,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Evidence(BaseModel):
    """A pointer into the trace justifying a score.

    Args:
        kind: What sort of evidence this is, e.g. ``assistant_text`` or ``tool_call``.
        turn_index: Where in the conversation it was found.
        excerpt: The literal text or serialised fragment.
        detail: Why this excerpt supports the verdict.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    turn_index: int | None = None
    excerpt: str = ""
    detail: str = ""


class EvaluationResult(BaseModel):
    """The verdict of one evaluator on one scenario, at turn or session level.

    Args:
        evaluation_id: Stable identifier.
        run_id: Run this belongs to.
        scenario_id: Scenario evaluated.
        session_id: Session evaluated.
        turn_id: Turn evaluated, for turn-level results.
        turn_index: Numeric turn position.
        evaluator_name: Which evaluator produced this.
        evaluator_kind: ``deterministic``, ``semantic`` or ``judge``. Kept separate so
            deterministic and model-graded scores are never silently averaged together.
        evaluation_level: Turn or session scope.
        dimension: Which quality axis this contributes to.
        score: Normalised 0..1 score.
        passed: Whether the check passed its own threshold.
        confidence: Evaluator self-reported confidence, 0..1.
        evidence: Trace pointers backing the verdict.
        reasoning_summary: One-line explanation for humans.
        failure_category: Taxonomy label when the check failed.
        severity: How much the failure matters.
        metadata: Evaluator-specific extras such as judge model and prompt version.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    run_id: str
    scenario_id: str
    session_id: str = ""
    turn_id: str | None = None
    turn_index: int | None = None
    evaluator_name: str
    evaluator_kind: str = "deterministic"
    evaluation_level: EvaluationLevel = EvaluationLevel.SESSION
    dimension: Dimension
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning_summary: str = ""
    failure_category: FailureCategory = FailureCategory.NONE
    severity: Severity = Severity.INFO
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # A plain property, deliberately not a computed_field. A computed field would be
    # written into model_dump_json() and then rejected on read by extra="forbid",
    # making every persisted result unloadable. It is cheaply re-derived anyway.
    @property
    def is_critical_failure(self) -> bool:
        """Whether this result alone blocks release.

        A result is release-blocking when it failed, its category is in the critical
        set, and it was marked critical severity. Requiring both guards against a
        low-severity heuristic accidentally gating a release.
        """
        return (
            not self.passed
            and self.severity is Severity.CRITICAL
            and is_critical(self.failure_category)
        )


class DimensionScore(BaseModel):
    """Aggregated score for one dimension of one session.

    Args:
        dimension: Which axis.
        score: Weighted mean of contributing results, 0..1.
        weight: Weight applied when rolling up to the overall score.
        result_count: How many results contributed.
        failed_count: How many of those failed.
    """

    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    result_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)


class SessionSummary(BaseModel):
    """Rolled-up scoring for one evaluated session.

    The individual dimension fields are duplicated as top-level attributes because
    analytics, the dashboard and the release report all consume them as a flat table;
    ``dimension_scores`` retains the structured form.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    session_id: str
    scenario_id: str
    scenario_category: str = ""
    scenario_difficulty: str = ""
    model_name: str = "mock"
    prompt_version: str = "v1"
    agent_version: str = "v1"
    turn_count: int = Field(default=0, ge=0)

    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    context_retention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    instruction_adherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)

    dimension_scores: list[DimensionScore] = Field(default_factory=list)

    total_latency: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)

    passed: bool = False
    critical_failures: list[str] = Field(default_factory=list)
    release_blockers: list[str] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    judge_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_critical_failure(self) -> bool:
        """Whether any release-blocking failure was recorded."""
        return bool(self.critical_failures)


class RunSummary(BaseModel):
    """Aggregate view of an entire evaluation run.

    Args:
        run_id: Identifier of the run.
        label: Human-friendly name, e.g. ``baseline`` or ``candidate``.
        suite_id: Scenario suite executed.
        provider_name: Provider that served the sessions.
        model_name: Model or behaviour profile under test.
        session_count: How many sessions were evaluated.
        metrics: Flat metric name to value mapping produced by the analytics layer.
        config_digest: Hash of the effective configuration, so a run can be tied to
            the exact rubric and thresholds that scored it.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    label: str = ""
    suite_id: str = ""
    provider_name: str = "mock"
    model_name: str = "mock"
    prompt_version: str = "v1"
    agent_version: str = "v1"
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    session_count: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    release_decision: ReleaseDecision = ReleaseDecision.FAIL
    config_digest: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThresholdCheck(BaseModel):
    """One release-gate comparison.

    Args:
        name: Metric checked.
        observed: Measured value.
        threshold: Required value.
        comparison: ``gte`` or ``lte``.
        passed: Whether the gate was satisfied.
        blocking: Whether failing it prevents release outright.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    observed: float
    threshold: float
    comparison: str = "gte"
    passed: bool = False
    blocking: bool = True
    note: str = ""


class ReleaseReadinessReport(BaseModel):
    """The decision artifact produced at the end of a run.

    Args:
        run_id: Run this report describes.
        decision: Pass, conditional pass or fail.
        executive_summary: Short prose verdict.
        threshold_checks: Every gate that was evaluated.
        release_blockers: Reasons release is blocked.
        remediation: Concrete recommended fixes.
        limitations: What this evaluation does *not* establish.
        result_label: Provenance banner, e.g. that results are simulated.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: datetime = Field(default_factory=_utc_now)
    decision: ReleaseDecision = ReleaseDecision.FAIL
    executive_summary: str = ""
    run_summary: RunSummary | None = None
    threshold_checks: list[ThresholdCheck] = Field(default_factory=list)
    scenario_composition: dict[str, int] = Field(default_factory=dict)
    metrics_by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    metrics_by_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    metrics_by_length: dict[str, dict[str, float]] = Field(default_factory=dict)
    tool_reliability: dict[str, float] = Field(default_factory=dict)
    critical_failures: list[dict[str, Any]] = Field(default_factory=list)
    regression_findings: list[str] = Field(default_factory=list)
    alignment_summary: dict[str, float] = Field(default_factory=dict)
    release_blockers: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    result_label: str = "Deterministic demonstration results using simulated model behavior"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricDelta(BaseModel):
    """Baseline-versus-candidate movement for one metric.

    Args:
        name: Metric name.
        baseline: Baseline value.
        candidate: Candidate value.
        absolute_change: candidate - baseline.
        relative_change: Fractional change against baseline, ``None`` when baseline is 0.
        tolerance: Allowed regression, expressed as a signed absolute change.
        regressed: Whether the movement breached tolerance.
        higher_is_better: Direction of improvement for this metric.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    baseline: float
    candidate: float
    absolute_change: float
    relative_change: float | None = None
    tolerance: float | None = None
    regressed: bool = False
    higher_is_better: bool = True


class ComparisonReport(BaseModel):
    """Structured result of comparing two runs.

    Args:
        baseline_run_id: The reference run.
        candidate_run_id: The run under scrutiny.
        deltas: Per-metric movement.
        regressions: Human-readable descriptions of tolerance breaches.
        gate_passed: Whether the regression gate allows the candidate through.
    """

    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    candidate_run_id: str
    generated_at: datetime = Field(default_factory=_utc_now)
    deltas: list[MetricDelta] = Field(default_factory=list)
    deltas_by_category: dict[str, list[MetricDelta]] = Field(default_factory=dict)
    deltas_by_difficulty: dict[str, list[MetricDelta]] = Field(default_factory=dict)
    deltas_by_length: dict[str, list[MetricDelta]] = Field(default_factory=dict)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    gate_passed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
