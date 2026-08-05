"""Statistics, aggregation, storage and reporting."""

from __future__ import annotations

import pytest
from tests.conftest import evaluate

from evalforge.analytics.alignment import (
    build_alignment_report,
    cohens_kappa,
    confusion_matrix,
    krippendorff_alpha,
    weighted_kappa,
)
from evalforge.analytics.metrics import bucket_length, compute_metrics
from evalforge.analytics.statistics import (
    MIN_N_FOR_INTERVAL,
    bootstrap_mean,
    cliffs_delta,
    cohens_h,
    interpret_kappa,
    spearman,
    wilson_interval,
)
from evalforge.config import EvalForgeConfig
from evalforge.evaluators.aggregation import aggregate_session, score_dimension
from evalforge.exceptions import RegressionGateError, RunNotFoundError
from evalforge.orchestration.comparison import compare_runs, enforce_gate
from evalforge.reporting.release_readiness import decide_release, evaluate_thresholds
from evalforge.schemas.annotation import DimensionRating, HumanAnnotation
from evalforge.schemas.common import (
    Dimension,
    FailureCategory,
    ReleaseDecision,
    Severity,
)
from evalforge.schemas.evaluation import EvaluationResult, RunSummary, SessionSummary
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace
from evalforge.storage.store import RunStore


def make_result(**overrides) -> EvaluationResult:
    """Build an evaluation result with sensible defaults."""
    payload = {
        "evaluation_id": "eva",
        "run_id": "run",
        "scenario_id": "scn",
        "session_id": "ses",
        "evaluator_name": "test",
        "dimension": Dimension.CONTEXT_RETENTION,
        "score": 1.0,
        "passed": True,
        "metadata": {"rollup": True},
    }
    payload.update(overrides)
    return EvaluationResult(**payload)


class TestWilsonInterval:
    """Wilson is used because the interesting rates sit near the boundaries."""

    def test_interval_brackets_the_estimate(self) -> None:
        interval = wilson_interval(85, 100)
        assert interval.lower < interval.point < interval.upper

    def test_stays_inside_zero_and_one_at_the_boundary(self) -> None:
        """The normal approximation would produce an upper bound above 1.0 here."""
        interval = wilson_interval(100, 100)
        assert interval.point == 1.0
        assert interval.upper <= 1.0
        assert interval.lower < 1.0

        interval = wilson_interval(0, 100)
        assert interval.lower >= 0.0

    def test_interval_is_suppressed_on_a_small_sample(self) -> None:
        interval = wilson_interval(3, MIN_N_FOR_INTERVAL - 1)
        assert interval.has_interval is False
        assert "suppressed" in interval.render()

    def test_empty_sample_is_safe(self) -> None:
        assert wilson_interval(0, 0).point == 0.0


class TestBootstrap:
    """Bootstrapping makes no distributional assumption, and must be reproducible."""

    def test_interval_brackets_the_mean(self) -> None:
        values = [0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.9, 0.91, 0.95, 0.99, 1.0, 0.88]
        interval = bootstrap_mean(values, seed=42)
        assert interval.lower < interval.point < interval.upper

    def test_is_reproducible(self) -> None:
        values = [0.1 * i for i in range(12)]
        assert bootstrap_mean(values, seed=7).lower == bootstrap_mean(values, seed=7).lower

    def test_suppressed_on_a_small_sample(self) -> None:
        assert bootstrap_mean([0.5, 0.6]).has_interval is False


class TestEffectSizes:
    """Effect sizes accompany deltas because a raw difference misleads at the edges."""

    def test_cohens_h_is_larger_near_the_boundary(self) -> None:
        """0.98 -> 0.95 is a bigger change than 0.50 -> 0.47, by the same 3 points."""
        near_edge = abs(cohens_h(0.95, 0.98))
        near_middle = abs(cohens_h(0.47, 0.50))
        assert near_edge > near_middle

    def test_cohens_h_is_zero_for_no_change(self) -> None:
        assert cohens_h(0.8, 0.8) == pytest.approx(0.0)

    def test_cliffs_delta_signs_correctly(self) -> None:
        assert cliffs_delta([0.9, 0.95, 1.0], [0.1, 0.2, 0.3]) == pytest.approx(1.0)
        assert cliffs_delta([0.1, 0.2, 0.3], [0.9, 0.95, 1.0]) == pytest.approx(-1.0)
        assert cliffs_delta([0.5], [0.5]) == pytest.approx(0.0)

    def test_spearman_captures_monotonic_relationships(self) -> None:
        assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_spearman_degenerates_safely(self) -> None:
        assert spearman([1, 1, 1], [1, 1, 1]) == 0.0
        assert spearman([1], [1]) == 0.0


class TestAgreementStatistics:
    """Chance correction is the whole point of kappa."""

    def test_kappa_is_near_zero_for_degenerate_raters(self) -> None:
        """Two raters who always say pass agree 100% of the time and know nothing."""
        assert cohens_kappa([True] * 20, [True] * 20) == pytest.approx(0.0)

    def test_kappa_is_one_for_perfect_informative_agreement(self) -> None:
        pattern = [True, False] * 10
        assert cohens_kappa(pattern, pattern) == pytest.approx(1.0)

    def test_kappa_is_negative_when_worse_than_chance(self) -> None:
        pattern = [True, False] * 10
        assert cohens_kappa(pattern, [not value for value in pattern]) < 0

    def test_weighted_kappa_penalises_distant_disagreement_more(self) -> None:
        """On a 1..5 scale, 4-vs-5 is not the same as 1-vs-5."""
        base = [1, 2, 3, 4, 5] * 4
        near = [1, 2, 3, 4, 4] * 4
        far = [5, 4, 3, 2, 1] * 4
        assert weighted_kappa(base, near) > weighted_kappa(base, far)

    def test_confusion_matrix_counts_all_four_cells(self) -> None:
        matrix = confusion_matrix([True, True, False, False], [True, False, True, False])
        assert matrix == {"true_pass": 1, "false_fail": 1, "false_pass": 1, "true_fail": 1}

    def test_krippendorff_needs_enough_overlap(self) -> None:
        assert krippendorff_alpha({"a": {"s1": 1.0}, "b": {"s1": 1.0}}) == 0.0

    def test_krippendorff_is_high_for_agreeing_raters(self) -> None:
        sessions = {f"s{i}": float(i % 5) / 4 for i in range(12)}
        alpha = krippendorff_alpha({"a": sessions, "b": dict(sessions)})
        assert alpha > 0.9

    def test_kappa_interpretations_are_labelled(self) -> None:
        assert interpret_kappa(0.9) == "almost perfect"
        assert interpret_kappa(0.7) == "substantial"
        assert interpret_kappa(-0.1) == "worse than chance"


class TestAggregation:
    """Scoring policy: penalties, and the critical-failure override."""

    def test_a_single_critical_failure_zeroes_its_dimension(self, config: EvalForgeConfig) -> None:
        results = [
            make_result(dimension=Dimension.SAFETY, score=1.0, passed=True),
            make_result(
                dimension=Dimension.SAFETY,
                score=0.0,
                passed=False,
                failure_category=FailureCategory.PROMPT_INJECTION_FOLLOWED,
                severity=Severity.CRITICAL,
                metadata={"rollup": False},
            ),
        ]
        score = score_dimension(Dimension.SAFETY, results, config)
        assert score.score == 0.0

    def test_failures_are_not_double_counted(self, config: EvalForgeConfig) -> None:
        """A per-failure result must be a penalty, not also a member of the basis.

        Counting it twice was a real bug: it dragged the mean down *and* subtracted a
        penalty for the same defect.
        """
        rollup_only = [make_result(score=0.8, passed=True, metadata={"rollup": True})]
        with_failure = [
            make_result(score=0.8, passed=True, metadata={"rollup": True}),
            make_result(
                score=0.0,
                passed=False,
                severity=Severity.MINOR,
                failure_category=FailureCategory.FACT_LOST,
                metadata={"rollup": False},
            ),
        ]
        clean = score_dimension(Dimension.CONTEXT_RETENTION, rollup_only, config)
        penalised = score_dimension(Dimension.CONTEXT_RETENTION, with_failure, config)
        expected_penalty = config.rubric.severity_penalties["minor"]
        assert penalised.score == pytest.approx(clean.score - expected_penalty, abs=1e-6)

    def test_critical_failure_fails_the_session_at_any_score(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        results.append(
            make_result(
                dimension=Dimension.SAFETY,
                score=0.0,
                passed=False,
                failure_category=FailureCategory.UNAUTHORIZED_ACTION,
                severity=Severity.CRITICAL,
                reasoning_summary="sent without approval",
                metadata={"rollup": False},
            )
        )
        summary = aggregate_session(demo_scenario, perfect_trace, results, config)
        assert summary.passed is False
        assert summary.critical_failures
        assert summary.release_blockers

    def test_judge_scores_stay_separate_from_deterministic_ones(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        summary = aggregate_session(demo_scenario, perfect_trace, results, config)
        assert summary.judge_scores, "judge results should be reported"
        assert "judge_failure_categories" in summary.metadata

    def test_session_records_the_config_digest(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        summary = aggregate_session(demo_scenario, perfect_trace, results, config)
        assert summary.metadata["config_digest"] == config.digest


class TestMetrics:
    """Metric computation over stored records."""

    def test_length_bucketing_snaps_to_the_analysis_grid(self) -> None:
        assert bucket_length(5) == 5
        assert bucket_length(12) == 10
        assert bucket_length(14) == 15
        assert bucket_length(99) == 30

    def test_metrics_are_empty_for_an_empty_run(self) -> None:
        metrics = compute_metrics("run", [], [])
        assert metrics.session_count == 0
        assert metrics.scalars == {}

    def test_subgroups_always_report_n(self) -> None:
        summaries = [
            SessionSummary(
                run_id="r",
                session_id=f"s{i}",
                scenario_id=f"scn{i}",
                scenario_category="goal_drift",
                scenario_difficulty="hard",
                turn_count=10,
                overall_score=0.8,
                passed=i % 2 == 0,
            )
            for i in range(6)
        ]
        metrics = compute_metrics("r", summaries, [])
        assert metrics.by_category["goal_drift"]["n"] == 6
        assert metrics.scalars["pass_rate"] == pytest.approx(0.5)


class TestReleaseGate:
    """The release decision, and its ordering."""

    def test_critical_failures_force_a_fail(self, config: EvalForgeConfig) -> None:
        summaries = [
            SessionSummary(
                run_id="r",
                session_id="s",
                scenario_id="scn",
                overall_score=1.0,
                passed=True,
                critical_failures=["prompt_injection_followed"],
            )
        ]
        metrics = compute_metrics("r", summaries, [])
        checks = evaluate_thresholds(metrics, config)
        decision, blockers = decide_release(checks, int(metrics.get("critical_failures")))
        assert decision is ReleaseDecision.FAIL
        assert any("critical" in blocker.lower() for blocker in blockers)

    def test_unmeasured_gates_are_reported_not_silently_passed(
        self, config: EvalForgeConfig
    ) -> None:
        metrics = compute_metrics("r", [], [])
        checks = evaluate_thresholds(metrics, config)
        unmeasured = [check for check in checks if "NOT MEASURED" in check.note]
        assert unmeasured, "a gate with no measurement must say so"
        assert all(not check.blocking for check in unmeasured)

    def test_advisory_failure_downgrades_to_conditional_pass(self) -> None:
        from evalforge.schemas.evaluation import ThresholdCheck

        checks = [
            ThresholdCheck(name="a", observed=1.0, threshold=0.9, passed=True, blocking=True),
            ThresholdCheck(name="b", observed=0.5, threshold=0.9, passed=False, blocking=False),
        ]
        decision, _ = decide_release(checks, 0)
        assert decision is ReleaseDecision.CONDITIONAL_PASS

    def test_all_gates_passing_is_a_pass(self) -> None:
        from evalforge.schemas.evaluation import ThresholdCheck

        checks = [ThresholdCheck(name="a", observed=1.0, threshold=0.9, passed=True, blocking=True)]
        decision, blockers = decide_release(checks, 0)
        assert decision is ReleaseDecision.PASS
        assert blockers == []


class TestRegressionGate:
    """Tolerance handling in both directions."""

    def _metrics(self, run_id: str, values: dict[str, float]):
        metrics = compute_metrics(run_id, [], [])
        metrics.scalars.update(values)
        return metrics

    def test_a_drop_beyond_tolerance_is_a_regression(self, config: EvalForgeConfig) -> None:
        baseline = self._metrics("base", {"context_retention": 0.95})
        candidate = self._metrics("cand", {"context_retention": 0.80})
        report = compare_runs(baseline, candidate, config, metrics=("context_retention",))
        assert report.gate_passed is False
        assert report.deltas[0].regressed is True
        with pytest.raises(RegressionGateError):
            enforce_gate(report)

    def test_a_drop_within_tolerance_passes(self, config: EvalForgeConfig) -> None:
        baseline = self._metrics("base", {"context_retention": 0.95})
        candidate = self._metrics("cand", {"context_retention": 0.94})
        report = compare_runs(baseline, candidate, config, metrics=("context_retention",))
        assert report.gate_passed is True

    def test_lower_is_better_metrics_flip_the_sign(self, config: EvalForgeConfig) -> None:
        """More critical failures is worse, even though the number went up."""
        baseline = self._metrics("base", {"critical_failure_count": 0.0})
        candidate = self._metrics("cand", {"critical_failure_count": 5.0})
        report = compare_runs(baseline, candidate, config, metrics=("critical_failure_count",))
        assert report.deltas[0].higher_is_better is False
        assert report.deltas[0].regressed is True

    def test_an_improvement_is_recorded(self, config: EvalForgeConfig) -> None:
        baseline = self._metrics("base", {"context_retention": 0.80})
        candidate = self._metrics("cand", {"context_retention": 0.95})
        report = compare_runs(baseline, candidate, config, metrics=("context_retention",))
        assert report.gate_passed is True
        assert report.improvements


class TestStorage:
    """Persistence round-trips and error handling."""

    def test_run_round_trip(self, store: RunStore) -> None:
        summary = RunSummary(run_id="run_x", label="baseline", metrics={"pass_rate": 0.9})
        store.save_run(summary)
        assert store.run_exists("run_x")
        assert store.get_run("run_x").metrics["pass_rate"] == pytest.approx(0.9)

    def test_missing_run_raises_a_helpful_error(self, store: RunStore) -> None:
        with pytest.raises(RunNotFoundError, match="not found"):
            store.get_run("nope")

    def test_trace_round_trip(self, store: RunStore, perfect_trace: SessionTrace) -> None:
        store.save_trace(perfect_trace)
        restored = store.get_trace(perfect_trace.run_id, perfect_trace.session_id)
        assert restored.session_id == perfect_trace.session_id
        assert restored.turn_count == perfect_trace.turn_count

    def test_evaluation_round_trip(
        self,
        store: RunStore,
        demo_scenario: Scenario,
        config: EvalForgeConfig,
        perfect_trace: SessionTrace,
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        store.save_run(RunSummary(run_id=perfect_trace.run_id))
        store.save_evaluations(results)
        restored = store.get_evaluations(perfect_trace.run_id)
        assert len(restored) == len(results)

    def test_sessions_are_returned_worst_first(self, store: RunStore) -> None:
        store.save_run(RunSummary(run_id="r"))
        for index, score in enumerate([0.9, 0.2, 0.6]):
            store.save_session(
                SessionSummary(
                    run_id="r",
                    session_id=f"s{index}",
                    scenario_id=f"scn{index}",
                    overall_score=score,
                )
            )
        scores = [item.overall_score for item in store.get_sessions("r")]
        assert scores == sorted(scores), "failures should surface first"

    def test_annotations_default_to_blind_only(self, store: RunStore) -> None:
        for index, blind in enumerate([True, False]):
            store.save_annotation(
                HumanAnnotation(
                    annotation_id=f"a{index}",
                    run_id="r",
                    session_id="s",
                    scenario_id="scn",
                    annotator_id=f"ann{index}",
                    blind=blind,
                )
            )
        assert len(store.get_annotations("r")) == 1
        assert len(store.get_annotations("r", blind_only=False)) == 2

    def test_scenarios_round_trip(self, store: RunStore, demo_scenario: Scenario) -> None:
        store.save_scenarios("r", [demo_scenario])
        restored = store.get_scenarios("r")
        assert len(restored) == 1
        assert restored[0].scenario_id == demo_scenario.scenario_id


class TestAlignmentReport:
    """The alignment analysis and its honesty guards."""

    def _annotations(self, count: int = 12) -> list[HumanAnnotation]:
        return [
            HumanAnnotation(
                annotation_id=f"a_{annotator}_{index}",
                run_id="r",
                session_id=f"s{index}",
                scenario_id=f"scn{index}",
                annotator_id=annotator,
                overall_pass=index % 3 != 0,
                ratings=[DimensionRating(dimension=Dimension.CONTEXT_RETENTION, rating=4)],
            )
            for annotator in ("annotator_a", "annotator_b")
            for index in range(count)
        ]

    def _summaries(self, count: int = 12) -> list[SessionSummary]:
        return [
            SessionSummary(
                run_id="r",
                session_id=f"s{index}",
                scenario_id=f"scn{index}",
                scenario_category="goal_drift",
                scenario_difficulty="hard",
                turn_count=10,
                overall_score=0.8,
                passed=index % 4 != 0,
            )
            for index in range(count)
        ]

    def test_report_computes_the_human_human_ceiling(self) -> None:
        report = build_alignment_report("r", self._annotations(), self._summaries())
        assert report.annotator_count == 2
        assert report.doubly_annotated == 12
        assert report.statistic("cohens_kappa", "annotator_a", "annotator_b") is not None

    def test_report_compares_against_each_automated_source(self) -> None:
        report = build_alignment_report("r", self._annotations(), self._summaries())
        pairs = {(item.rater_a, item.rater_b) for item in report.statistics}
        assert ("human", "aggregate") in pairs
        assert ("human", "judge") in pairs or ("human", "deterministic") in pairs

    def test_no_annotations_is_reported_as_a_limitation(self) -> None:
        report = build_alignment_report("r", [], self._summaries())
        assert report.annotation_count == 0
        assert report.limitations, "an unverified evaluator must say so"

    def test_single_annotator_is_flagged(self) -> None:
        single = [item for item in self._annotations() if item.annotator_id == "annotator_a"]
        report = build_alignment_report("r", single, self._summaries())
        assert any("one annotator" in item for item in report.limitations)

    def test_non_blind_annotations_are_excluded(self) -> None:
        contaminated = [item.model_copy(update={"blind": False}) for item in self._annotations()]
        report = build_alignment_report("r", contaminated, self._summaries())
        assert report.annotation_count == 0
