"""Human-versus-automated evaluator alignment.

This module answers the question that decides whether the rest of EvalForge means
anything: *do the automated evaluators measure what a careful human would?*

Four comparisons are computed. Human-versus-human first, because it establishes the
ceiling — no automated evaluator should be expected to agree with humans more than
humans agree with each other, and an alignment number reported without that ceiling is
uninterpretable.

All logic lives here rather than in the notebook. The notebook demonstrates the
analysis; it is not the implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from evalforge.analytics.metrics import bucket_length
from evalforge.analytics.statistics import interpret_kappa, rate, spearman
from evalforge.schemas.annotation import (
    AgreementStatistic,
    AlignmentReport,
    HumanAnnotation,
)
from evalforge.schemas.common import Dimension
from evalforge.schemas.evaluation import SessionSummary

#: Dimensions compared between humans and automation.
COMPARED_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension.TASK_COMPLETION,
    Dimension.CONTEXT_RETENTION,
    Dimension.INSTRUCTION_ADHERENCE,
    Dimension.RECOVERY_QUALITY,
)

#: Minimum paired observations before a statistic is reported at all.
MIN_PAIRS = 5


@dataclass
class PairedVerdicts:
    """Aligned pass/fail and score pairs for two raters."""

    rater_a: str
    rater_b: str
    a_pass: list[bool]
    b_pass: list[bool]
    a_score: list[float]
    b_score: list[float]
    session_ids: list[str]

    def __len__(self) -> int:
        """Number of paired observations."""
        return len(self.a_pass)


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters.

    Chance correction matters enormously here. If 85% of sessions pass, two raters who
    both always say "pass" show 85% raw agreement and kappa near zero — and the kappa is
    the honest number.
    """
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        # Both raters are degenerate (all one class). Perfect agreement is then
        # uninformative rather than perfect, so report 0.0 rather than a spurious 1.0.
        return 0.0
    return (observed - expected) / (1 - expected)


def weighted_kappa(a: list[int], b: list[int], categories: int = 5) -> float:
    """Quadratically weighted kappa for ordinal ratings.

    On a 1..5 rubric, a 4-vs-5 disagreement is not the same as a 1-vs-5. Quadratic
    weighting penalises distant disagreements more, which is the behaviour a rubric
    scale actually wants.
    """
    if len(a) != len(b) or not a:
        return 0.0

    observed = np.zeros((categories, categories), dtype=float)
    for x, y in zip(a, b, strict=True):
        observed[_clamp(x, categories), _clamp(y, categories)] += 1
    observed /= observed.sum()

    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b)

    indices = np.arange(categories)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (categories - 1) ** 2

    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 0.0
    return 1.0 - float((weights * observed).sum()) / denominator


def _clamp(value: int, categories: int) -> int:
    """Clamp a 1-based rating into a 0-based index."""
    return max(0, min(categories - 1, value - 1))


def krippendorff_alpha(ratings: dict[str, dict[str, float]]) -> float:
    """Krippendorff's alpha for interval data with missing observations.

    Used in addition to kappa because kappa handles exactly two raters and no missing
    data, neither of which survives contact with a real annotation exercise where
    annotators cover overlapping-but-different subsets.

    Args:
        ratings: ``{annotator_id: {session_id: value}}``.

    Returns:
        Alpha, or 0.0 when fewer than two annotators overlap on enough units — reported
        as unavailable rather than computed on an inadequate sample.
    """
    units: dict[str, list[float]] = defaultdict(list)
    for per_session in ratings.values():
        for session_id, value in per_session.items():
            units[session_id].append(value)

    usable = {unit: values for unit, values in units.items() if len(values) >= 2}
    if len(usable) < MIN_PAIRS:
        return 0.0

    # Observed disagreement: mean squared difference within units.
    observed_pairs: list[float] = []
    for values in usable.values():
        for left, right in combinations(values, 2):
            observed_pairs.append((left - right) ** 2)
    if not observed_pairs:
        return 0.0
    observed = float(np.mean(observed_pairs))

    # Expected disagreement: mean squared difference across all values regardless of unit.
    everything = [value for values in usable.values() for value in values]
    expected_pairs = [(left - right) ** 2 for left, right in combinations(everything, 2)]
    if not expected_pairs:
        return 0.0
    expected = float(np.mean(expected_pairs))

    if expected == 0:
        return 0.0
    return 1.0 - observed / expected


def confusion_matrix(a: list[bool], b: list[bool]) -> dict[str, int]:
    """2x2 pass/fail confusion counts between two raters.

    ``rater_a`` is treated as reference, so ``false_positive`` means "b passed a session
    a failed" — the direction that matters when b is an automated evaluator, because a
    false pass ships a broken agent.
    """
    matrix = {"true_pass": 0, "true_fail": 0, "false_pass": 0, "false_fail": 0}
    for reference, candidate in zip(a, b, strict=True):
        if reference and candidate:
            matrix["true_pass"] += 1
        elif not reference and not candidate:
            matrix["true_fail"] += 1
        elif not reference and candidate:
            matrix["false_pass"] += 1
        else:
            matrix["false_fail"] += 1
    return matrix


def _human_pass(annotation: HumanAnnotation) -> bool:
    """The annotator's holistic verdict."""
    return annotation.overall_pass


def _human_score(annotation: HumanAnnotation) -> float:
    """Mean normalised rubric rating for one annotation."""
    if not annotation.ratings:
        return 1.0 if annotation.overall_pass else 0.0
    return float(np.mean([rating.normalized for rating in annotation.ratings]))


def pair_human_with_human(
    annotations: list[HumanAnnotation],
) -> list[PairedVerdicts]:
    """Build paired verdicts for every pair of annotators with shared sessions."""
    by_annotator: dict[str, dict[str, HumanAnnotation]] = defaultdict(dict)
    for annotation in annotations:
        by_annotator[annotation.annotator_id][annotation.session_id] = annotation

    paired: list[PairedVerdicts] = []
    for left, right in combinations(sorted(by_annotator), 2):
        shared = sorted(set(by_annotator[left]) & set(by_annotator[right]))
        if len(shared) < MIN_PAIRS:
            continue
        paired.append(
            PairedVerdicts(
                rater_a=left,
                rater_b=right,
                a_pass=[_human_pass(by_annotator[left][s]) for s in shared],
                b_pass=[_human_pass(by_annotator[right][s]) for s in shared],
                a_score=[_human_score(by_annotator[left][s]) for s in shared],
                b_score=[_human_score(by_annotator[right][s]) for s in shared],
                session_ids=shared,
            )
        )
    return paired


def pair_human_with_automated(
    annotations: list[HumanAnnotation],
    summaries: list[SessionSummary],
    automated: str = "aggregate",
) -> PairedVerdicts:
    """Pair every blind annotation with the automated verdict for the same session.

    Args:
        annotations: Blind human annotations.
        summaries: Session summaries carrying the automated verdicts.
        automated: ``aggregate`` for the overall score, ``judge`` for judge scores only,
            or ``deterministic`` for the pass flag from deterministic checks.
    """
    by_session = {summary.session_id: summary for summary in summaries}
    a_pass: list[bool] = []
    b_pass: list[bool] = []
    a_score: list[float] = []
    b_score: list[float] = []
    session_ids: list[str] = []

    for annotation in sorted(annotations, key=lambda item: item.session_id):
        summary = by_session.get(annotation.session_id)
        if summary is None:
            continue
        if automated == "judge":
            if not summary.judge_scores:
                continue
            score = float(np.mean(list(summary.judge_scores.values())))
            verdict = score >= 0.7
        elif automated == "deterministic":
            score = summary.overall_score
            verdict = not summary.critical_failures and score >= 0.75
        else:
            score = summary.overall_score
            verdict = summary.passed

        a_pass.append(_human_pass(annotation))
        b_pass.append(verdict)
        a_score.append(_human_score(annotation))
        b_score.append(score)
        session_ids.append(annotation.session_id)

    return PairedVerdicts(
        rater_a="human",
        rater_b=automated,
        a_pass=a_pass,
        b_pass=b_pass,
        a_score=a_score,
        b_score=b_score,
        session_ids=session_ids,
    )


def _statistic(pair: PairedVerdicts, dimension: str | None = None) -> list[AgreementStatistic]:
    """Compute the standard statistic set for one rater pair."""
    if len(pair) < MIN_PAIRS:
        return []

    raw = rate(sum(1 for x, y in zip(pair.a_pass, pair.b_pass, strict=True) if x == y), len(pair))
    kappa = cohens_kappa(pair.a_pass, pair.b_pass)
    correlation = spearman(pair.a_score, pair.b_score)

    return [
        AgreementStatistic(
            name="raw_agreement",
            rater_a=pair.rater_a,
            rater_b=pair.rater_b,
            dimension=dimension,
            value=round(raw, 4),
            n=len(pair),
            interpretation=f"{raw:.0%} of verdicts matched before chance correction",
        ),
        AgreementStatistic(
            name="cohens_kappa",
            rater_a=pair.rater_a,
            rater_b=pair.rater_b,
            dimension=dimension,
            value=round(kappa, 4),
            n=len(pair),
            interpretation=interpret_kappa(kappa),
        ),
        AgreementStatistic(
            name="spearman_rho",
            rater_a=pair.rater_a,
            rater_b=pair.rater_b,
            dimension=dimension,
            value=round(correlation, 4),
            n=len(pair),
            interpretation="rank correlation of scores; ordinal by design",
        ),
    ]


def _subgroup_agreement(
    annotations: list[HumanAnnotation],
    summaries: list[SessionSummary],
    key: Any,
) -> dict[str, float]:
    """Human/automated raw agreement within each subgroup."""
    by_session = {summary.session_id: summary for summary in summaries}
    grouped: dict[str, list[bool]] = defaultdict(list)

    for annotation in annotations:
        summary = by_session.get(annotation.session_id)
        if summary is None:
            continue
        grouped[str(key(summary))].append(_human_pass(annotation) == summary.passed)

    return {
        name: round(rate(sum(values), len(values)), 4)
        for name, values in sorted(grouped.items())
        if values
    }


def analyse_biases(
    annotations: list[HumanAnnotation],
    summaries: list[SessionSummary],
    verbosity: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Detect systematic biases in the automated evaluators.

    Args:
        annotations: Blind human annotations.
        summaries: Session summaries.
        verbosity: ``{session_id: assistant_character_count}``, used for the verbosity
            and conciseness analyses. Omitted when unavailable rather than approximated.
    """
    by_session = {summary.session_id: summary for summary in summaries}
    findings: list[dict[str, Any]] = []

    human_scores: list[float] = []
    auto_scores: list[float] = []
    lengths: list[float] = []
    turn_counts: list[float] = []

    for annotation in annotations:
        summary = by_session.get(annotation.session_id)
        if summary is None:
            continue
        human_scores.append(_human_score(annotation))
        auto_scores.append(summary.overall_score)
        turn_counts.append(float(summary.turn_count))
        if verbosity:
            lengths.append(float(verbosity.get(annotation.session_id, 0)))

    if verbosity and len(lengths) >= MIN_PAIRS:
        judge_by_length = spearman(lengths, auto_scores)
        human_by_length = spearman(lengths, human_scores)
        findings.append(
            {
                "bias": "verbosity",
                "automated_correlation_with_length": round(judge_by_length, 4),
                "human_correlation_with_length": round(human_by_length, 4),
                "residual": round(judge_by_length - human_by_length, 4),
                "interpretation": (
                    "A positive residual means the automated score rewards length more "
                    "than humans do, which is the classic judge failure mode."
                ),
                "n": len(lengths),
            }
        )

        concise = [
            summary.overall_score
            for annotation in annotations
            if (summary := by_session.get(annotation.session_id)) is not None
            and verbosity.get(annotation.session_id, 0) < float(np.median(lengths))
            and not summary.critical_failures
        ]
        verbose = [
            summary.overall_score
            for annotation in annotations
            if (summary := by_session.get(annotation.session_id)) is not None
            and verbosity.get(annotation.session_id, 0) >= float(np.median(lengths))
            and not summary.critical_failures
        ]
        if concise and verbose:
            findings.append(
                {
                    "bias": "over_penalisation_of_concise_answers",
                    "mean_score_concise": round(float(np.mean(concise)), 4),
                    "mean_score_verbose": round(float(np.mean(verbose)), 4),
                    "gap": round(float(np.mean(verbose)) - float(np.mean(concise)), 4),
                    "interpretation": (
                        "Computed over sessions with no critical failure, so a positive "
                        "gap reflects style rather than correctness."
                    ),
                    "n": len(concise) + len(verbose),
                }
            )

    if len(turn_counts) >= MIN_PAIRS:
        agreement_by_length: dict[str, list[bool]] = defaultdict(list)
        for annotation in annotations:
            summary = by_session.get(annotation.session_id)
            if summary is None:
                continue
            bucket = str(bucket_length(summary.turn_count))
            agreement_by_length[bucket].append(_human_pass(annotation) == summary.passed)

        by_bucket = {
            bucket: round(rate(sum(values), len(values)), 4)
            for bucket, values in sorted(agreement_by_length.items(), key=lambda kv: int(kv[0]))
        }
        if len(by_bucket) >= 2:
            buckets = sorted(by_bucket, key=int)
            findings.append(
                {
                    "bias": "reduced_reliability_on_long_sessions",
                    "agreement_by_length": by_bucket,
                    "shortest": by_bucket[buckets[0]],
                    "longest": by_bucket[buckets[-1]],
                    "drop": round(by_bucket[buckets[0]] - by_bucket[buckets[-1]], 4),
                    "interpretation": (
                        "A positive drop supports the hypothesis that evaluator "
                        "reliability decays as context length grows."
                    ),
                }
            )

    drift_sessions = [
        annotation
        for annotation in annotations
        if (summary := by_session.get(annotation.session_id)) is not None
        and summary.scenario_category == "goal_drift"
    ]
    if len(drift_sessions) >= MIN_PAIRS:
        agreements = [
            _human_pass(annotation) == by_session[annotation.session_id].passed
            for annotation in drift_sessions
        ]
        findings.append(
            {
                "bias": "subtle_goal_drift_detection",
                "agreement_on_goal_drift": round(rate(sum(agreements), len(agreements)), 4),
                "n": len(agreements),
                "interpretation": (
                    "Goal drift is the category humans and automated checks are both "
                    "expected to be weakest on; low agreement here is expected, and is "
                    "the reason the drift evaluator is deliberately conservative."
                ),
            }
        )

    # Position bias: does the turn index of the first failure predict the score?
    first_failure_turns: list[float] = []
    scores_for_position: list[float] = []
    for summary in summaries:
        position = summary.metadata.get("first_failure_turn")
        if isinstance(position, int | float):
            first_failure_turns.append(float(position))
            scores_for_position.append(summary.overall_score)
    if len(first_failure_turns) >= MIN_PAIRS:
        findings.append(
            {
                "bias": "position",
                "correlation_first_failure_turn_with_score": round(
                    spearman(first_failure_turns, scores_for_position), 4
                ),
                "n": len(first_failure_turns),
                "interpretation": (
                    "Near-zero means the score does not depend on *where* in the session "
                    "a failure occurred, which is the intended behaviour."
                ),
            }
        )

    return findings


def disagreement_examples(
    annotations: list[HumanAnnotation],
    summaries: list[SessionSummary],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Sessions where the human and automated verdicts diverge.

    Sorted by score gap, so the most informative disagreements come first — these are
    the cases worth adjudicating against the trace evidence.
    """
    by_session = {summary.session_id: summary for summary in summaries}
    rows: list[dict[str, Any]] = []

    for annotation in annotations:
        summary = by_session.get(annotation.session_id)
        if summary is None or _human_pass(annotation) == summary.passed:
            continue
        human_score = _human_score(annotation)
        rows.append(
            {
                "session_id": annotation.session_id,
                "scenario_id": annotation.scenario_id,
                "category": summary.scenario_category,
                "difficulty": summary.scenario_difficulty,
                "turn_count": summary.turn_count,
                "human_pass": annotation.overall_pass,
                "automated_pass": summary.passed,
                "human_score": round(human_score, 4),
                "automated_score": round(summary.overall_score, 4),
                "gap": round(abs(human_score - summary.overall_score), 4),
                "annotator": annotation.annotator_id,
                "human_categories": [c.value for c in annotation.failure_categories],
                "automated_categories": summary.failure_categories,
                "comment": annotation.comments,
            }
        )

    return sorted(rows, key=lambda row: -float(row["gap"]))[:limit]


def build_alignment_report(
    run_id: str,
    annotations: list[HumanAnnotation],
    summaries: list[SessionSummary],
    verbosity: dict[str, int] | None = None,
) -> AlignmentReport:
    """Compute the full alignment analysis for one run.

    Args:
        run_id: Run being analysed.
        annotations: Blind human annotations only.
        summaries: Session summaries for the same run.
        verbosity: Optional assistant output lengths for the bias analyses.
    """
    blind = [annotation for annotation in annotations if annotation.blind]
    report = AlignmentReport(
        run_id=run_id,
        annotation_count=len(blind),
        annotator_count=len({annotation.annotator_id for annotation in blind}),
    )

    session_counts: dict[str, int] = defaultdict(int)
    for annotation in blind:
        session_counts[annotation.session_id] += 1
    report.doubly_annotated = sum(1 for count in session_counts.values() if count >= 2)

    if not blind:
        report.limitations.append(
            "No blind annotations are available, so no agreement statistic could be "
            "computed. Collect annotations with `evalforge annotate`."
        )
        return report

    # 1. Human vs human — the ceiling.
    for pair in pair_human_with_human(blind):
        report.statistics.extend(_statistic(pair))
        report.confusion_matrices[f"{pair.rater_a}_vs_{pair.rater_b}"] = confusion_matrix(
            pair.a_pass, pair.b_pass
        )

    # 2-4. Human vs each automated source.
    for automated in ("deterministic", "judge", "aggregate"):
        pair = pair_human_with_automated(blind, summaries, automated)
        report.statistics.extend(_statistic(pair))
        if len(pair) >= MIN_PAIRS:
            report.confusion_matrices[f"human_vs_{automated}"] = confusion_matrix(
                pair.a_pass, pair.b_pass
            )

    # Ordinal agreement on rubric dimensions.
    for dimension in COMPARED_DIMENSIONS:
        ratings = _dimension_ratings(blind, dimension)
        if len(ratings) < 2:
            continue
        annotators = sorted(ratings)
        for left, right in combinations(annotators, 2):
            shared = sorted(set(ratings[left]) & set(ratings[right]))
            if len(shared) < MIN_PAIRS:
                continue
            value = weighted_kappa(
                [ratings[left][s] for s in shared], [ratings[right][s] for s in shared]
            )
            report.statistics.append(
                AgreementStatistic(
                    name="weighted_kappa",
                    rater_a=left,
                    rater_b=right,
                    dimension=dimension.value,
                    value=round(value, 4),
                    n=len(shared),
                    interpretation=interpret_kappa(value),
                )
            )

    # Krippendorff's alpha over the holistic score.
    alpha_input: dict[str, dict[str, float]] = defaultdict(dict)
    for annotation in blind:
        alpha_input[annotation.annotator_id][annotation.session_id] = _human_score(annotation)
    alpha = krippendorff_alpha(alpha_input)
    if alpha:
        report.statistics.append(
            AgreementStatistic(
                name="krippendorff_alpha",
                rater_a="all_annotators",
                rater_b="all_annotators",
                value=round(alpha, 4),
                n=report.doubly_annotated,
                interpretation=interpret_kappa(alpha),
            )
        )
    else:
        report.limitations.append(
            "Krippendorff's alpha was not computed: too few sessions carry two or more "
            "independent annotations to support it."
        )

    report.agreement_by_length = _subgroup_agreement(
        blind, summaries, lambda s: bucket_length(s.turn_count)
    )
    report.agreement_by_difficulty = _subgroup_agreement(
        blind, summaries, lambda s: s.scenario_difficulty
    )
    report.agreement_by_failure_type = _subgroup_agreement(
        blind, summaries, lambda s: s.scenario_category
    )
    report.bias_findings = analyse_biases(blind, summaries, verbosity)
    report.disagreement_examples = disagreement_examples(blind, summaries)

    if report.annotator_count < 2:
        report.limitations.append(
            "Only one annotator contributed, so no human-versus-human ceiling could be "
            "established and the automated agreement figure has nothing to be judged "
            "against."
        )
    if len(blind) < 30:
        report.limitations.append(
            f"Only {len(blind)} blind annotations were available; agreement statistics "
            "at this sample size have wide uncertainty and should be read as indicative."
        )

    return report


def _dimension_ratings(
    annotations: list[HumanAnnotation], dimension: Dimension
) -> dict[str, dict[str, int]]:
    """Ordinal ratings for one dimension, keyed by annotator and session."""
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for annotation in annotations:
        rating = annotation.rating_for(dimension)
        if rating is not None:
            output[annotation.annotator_id][annotation.session_id] = rating.rating
    return output
