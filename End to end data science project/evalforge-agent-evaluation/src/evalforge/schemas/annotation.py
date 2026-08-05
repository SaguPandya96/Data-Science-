"""Human annotation schema.

Human labels are the only external check on whether the automated evaluators measure
what they claim to. The schema records *when* an annotator submitted relative to seeing
automated scores, because an annotation collected after the fact is contaminated and
must not be pooled with blind ones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalforge.schemas.common import Dimension, FailureCategory, Severity


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DimensionRating(BaseModel):
    """One annotator's rating of one dimension.

    Args:
        dimension: Axis being rated.
        rating: Ordinal 1..5 rubric score. An ordinal scale is used rather than a
            continuous one because humans are not reliable at fine-grained continuous
            judgement, and it lets weighted kappa treat near-misses as near-misses.
        comment: Optional free-text justification.
    """

    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    rating: int = Field(ge=1, le=5)
    comment: str = ""

    @property
    def normalized(self) -> float:
        """Rating mapped onto the 0..1 scale used by automated evaluators."""
        return (self.rating - 1) / 4.0


class HumanAnnotation(BaseModel):
    """A complete blind annotation of one session by one annotator.

    Args:
        annotation_id: Stable identifier.
        run_id: Run the annotated session belongs to.
        session_id: Session annotated.
        scenario_id: Scenario the session executed.
        annotator_id: Pseudonymous annotator label, e.g. ``annotator_a``.
        ratings: Per-dimension ordinal ratings.
        overall_pass: The annotator's holistic pass/fail verdict.
        failure_categories: Taxonomy labels the annotator applied.
        severity: Worst severity the annotator observed.
        comments: Free-text notes.
        blind: Whether automated scores were hidden at submission time. Only blind
            annotations are used for agreement statistics.
        duration_seconds: How long the annotation took, used to flag rushed labels.
    """

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    run_id: str
    session_id: str
    scenario_id: str
    annotator_id: str
    ratings: list[DimensionRating] = Field(default_factory=list)
    overall_pass: bool = True
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    severity: Severity = Severity.INFO
    comments: str = ""
    blind: bool = True
    duration_seconds: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def rating_for(self, dimension: Dimension) -> DimensionRating | None:
        """Return this annotator's rating for ``dimension``, if given."""
        for rating in self.ratings:
            if rating.dimension is dimension:
                return rating
        return None

    def normalized_scores(self) -> dict[str, float]:
        """All ratings on the 0..1 scale, keyed by dimension name."""
        return {r.dimension.value: r.normalized for r in self.ratings}


class AgreementStatistic(BaseModel):
    """One computed agreement measure between two rater sources.

    Args:
        name: Statistic name, e.g. ``cohens_kappa``.
        rater_a: Label of the first rater source.
        rater_b: Label of the second.
        dimension: Dimension the statistic covers, or ``None`` for the holistic verdict.
        value: The statistic's value.
        n: Number of paired observations.
        interpretation: Plain-language reading of the value.
        confidence_interval: 95% interval where one could be computed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rater_a: str
    rater_b: str
    dimension: str | None = None
    value: float
    n: int = Field(default=0, ge=0)
    interpretation: str = ""
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlignmentReport(BaseModel):
    """Everything the alignment analysis produced for one run.

    Args:
        run_id: Run analysed.
        annotation_count: Number of blind annotations available.
        annotator_count: Distinct annotators.
        doubly_annotated: Sessions carrying at least two independent annotations.
        statistics: All computed agreement measures.
        confusion_matrices: Rater-pair name to 2x2 pass/fail counts.
        bias_findings: Detected systematic biases with effect sizes.
        disagreement_examples: Sessions where human and automated verdicts diverge.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: datetime = Field(default_factory=_utc_now)
    annotation_count: int = 0
    annotator_count: int = 0
    doubly_annotated: int = 0
    statistics: list[AgreementStatistic] = Field(default_factory=list)
    confusion_matrices: dict[str, dict[str, int]] = Field(default_factory=dict)
    agreement_by_failure_type: dict[str, float] = Field(default_factory=dict)
    agreement_by_length: dict[str, float] = Field(default_factory=dict)
    agreement_by_difficulty: dict[str, float] = Field(default_factory=dict)
    bias_findings: list[dict[str, Any]] = Field(default_factory=list)
    disagreement_examples: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def statistic(self, name: str, rater_a: str, rater_b: str) -> AgreementStatistic | None:
        """Look up one statistic by name and rater pair."""
        for stat in self.statistics:
            if stat.name == name and stat.rater_a == rater_a and stat.rater_b == rater_b:
                return stat
        return None
