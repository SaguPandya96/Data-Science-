"""Evaluator behaviour, including the false-positive and false-negative controls.

The two most important tests here are ``test_perfect_agent_has_no_failures`` and
``test_broken_agent_is_caught``. Together they bound the evaluators from both sides: an
evaluator that fires on a flawless session is unusable, and one that stays silent on a
pathological session is worse than nothing.
"""

from __future__ import annotations

import pytest
from tests.conftest import evaluate, run_scenario

from evalforge.config import EvalForgeConfig
from evalforge.evaluators.aggregation import aggregate_session, critical_failures
from evalforge.evaluators.base import (
    dates_match,
    fact_matches,
    lcs_ratio,
    normalize_money,
    normalize_text,
)
from evalforge.evaluators.context import DateAccuracyEvaluator, FactRetentionEvaluator
from evalforge.evaluators.judge import JudgeEvaluator, MockJudge
from evalforge.evaluators.registry import DETERMINISTIC_EVALUATORS, build_evaluators
from evalforge.evaluators.semantic import LexicalEmbedder, SemanticEvaluator
from evalforge.evaluators.tool_use import arguments_match
from evalforge.schemas.common import FailureCategory, Severity
from evalforge.schemas.scenario import Fact, FactKind, Scenario
from evalforge.schemas.trace import SessionTrace


class TestComparisonPrimitives:
    """Normalisation is policy: it must have one answer, everywhere."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("2026-09-15", "2026-09-15"),
            ("2026-09-15", "September 15"),
            ("September 15", "the launch is on September 15"),
        ],
    )
    def test_dates_match_across_forms(self, left: str, right: str) -> None:
        assert dates_match(left, right) is True

    def test_different_dates_do_not_match(self) -> None:
        assert dates_match("2026-09-15", "2026-09-16") is False
        assert dates_match("2026-09-15", "October 1") is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("$15,000", 15000.0), ("15000", 15000.0), ("15000.50 USD", 15000.5), (20000, 20000.0)],
    )
    def test_money_normalisation(self, value: object, expected: float) -> None:
        assert normalize_money(value) == pytest.approx(expected)

    def test_money_returns_none_when_absent(self) -> None:
        """Absent and zero are different failures and must stay distinguishable."""
        assert normalize_money("no figure here") is None
        assert normalize_money(None) is None
        assert normalize_money(0) == 0.0

    def test_text_normalisation_collapses_whitespace_and_case(self) -> None:
        assert normalize_text("  Analytics   Dashboard \n") == "analytics dashboard"

    def test_fact_matching_respects_kind(self) -> None:
        money = Fact(key="budget", value="15000", kind=FactKind.MONEY, turn_introduced=0)
        assert fact_matches(money, "$15,000") is True
        assert fact_matches(money, "15001") is False

        date = Fact(
            key="launch_date",
            value="2026-09-15",
            kind=FactKind.DATE,
            turn_introduced=0,
            aliases=("September 15",),
        )
        assert fact_matches(date, "September 15") is True
        assert fact_matches(date, "2026-10-01") is False

    def test_lcs_ratio_rewards_order_without_demanding_equality(self) -> None:
        assert lcs_ratio(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
        assert lcs_ratio(["a", "b", "c"], ["a", "x", "b", "c"]) == pytest.approx(1.0)
        assert lcs_ratio(["a", "b", "c"], ["c", "b", "a"]) < 1.0
        assert lcs_ratio([], ["a"]) == pytest.approx(1.0)

    def test_argument_matching_compares_only_pinned_keys(self) -> None:
        ok, mismatched = arguments_match({"budget": 15000.0}, {"budget": 15000, "extra": "ignored"})
        assert ok is True
        assert mismatched == []

        ok, mismatched = arguments_match({"budget": 15000.0}, {"budget": 20000})
        assert ok is False
        assert mismatched == ["budget"]

    def test_argument_matching_treats_dates_semantically(self) -> None:
        ok, _ = arguments_match({"launch_date": "2026-09-15"}, {"launch_date": "September 15"})
        assert ok is True


class TestFalsePositiveControl:
    """A flawless session must produce no failures anywhere."""

    def test_perfect_agent_has_no_failures(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        failures = [
            item for item in results if not item.passed and item.evaluator_kind == "deterministic"
        ]
        assert failures == [], f"false positives: {[f.evaluator_name for f in failures]}"

    def test_perfect_agent_scores_at_the_top(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, perfect_trace, config)
        summary = aggregate_session(demo_scenario, perfect_trace, results, config)
        assert summary.passed is True
        assert summary.overall_score > 0.95
        assert summary.critical_failures == []

    def test_perfect_agent_retains_every_fact(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        assert perfect_trace.final_workspace is not None
        facts = perfect_trace.final_workspace.facts
        assert facts["launch_date"] == "2026-09-15"
        assert facts["budget"] == "15000", "the turn-2 revision must be the value held"


class TestFalseNegativeControl:
    """A pathological session must be caught, loudly."""

    def test_broken_agent_is_caught(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, broken_trace, config)
        summary = aggregate_session(demo_scenario, broken_trace, results, config)
        assert summary.passed is False
        assert summary.overall_score < 0.7
        assert summary.failure_categories, "a broken session must be labelled"

    def test_broken_agent_loses_context(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = FactRetentionEvaluator(config).evaluate(demo_scenario, broken_trace)
        rollup = next(item for item in results if item.metadata.get("rollup"))
        assert rollup.score < 1.0
        assert rollup.metadata["lost"] or rollup.metadata["corrupted"]

    def test_broken_agent_violates_the_content_ban(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, broken_trace, config)
        categories = {item.failure_category for item in results if not item.passed}
        assert FailureCategory.FORBIDDEN_CONTENT in categories


class TestEvidenceRequirement:
    """A verdict with no pointer into the trace is unauditable."""

    def test_every_failure_carries_a_reason(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, broken_trace, config)
        for item in results:
            if not item.passed:
                assert item.reasoning_summary, f"{item.evaluator_name} failed without a reason"

    def test_critical_failures_always_carry_evidence(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, broken_trace, config)
        for item in critical_failures(results):
            assert item.evidence, f"{item.evaluator_name} blocked release without evidence"


class TestSeverityPolicy:
    """Only exact checks may gate a release."""

    def test_no_judge_result_is_ever_critical(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        """ADR-004: a categorical blocker must rest on an exact check, not an opinion."""
        results = evaluate(demo_scenario, broken_trace, config)
        judged = [item for item in results if item.evaluator_kind == "judge"]
        assert judged, "the judge should have produced results"
        assert all(item.severity is not Severity.CRITICAL for item in judged)

    def test_no_semantic_result_is_ever_a_failure(
        self, demo_scenario: Scenario, config: EvalForgeConfig, broken_trace: SessionTrace
    ) -> None:
        results = evaluate(demo_scenario, broken_trace, config)
        semantic = [item for item in results if item.evaluator_kind == "semantic"]
        assert semantic, "semantic evaluators should have run"
        assert all(item.passed for item in semantic)

    def test_date_loss_is_critical(self, config: EvalForgeConfig) -> None:
        """A dropped deadline is categorical, not statistical."""
        assert DateAccuracyEvaluator(config).name == "date_accuracy"


class TestDeterminism:
    """The same inputs must always produce the same verdict."""

    def test_evaluation_is_repeatable(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        first = evaluate(demo_scenario, perfect_trace, config)
        second = evaluate(demo_scenario, perfect_trace, config)
        assert [(r.evaluator_name, r.score) for r in first] == [
            (r.evaluator_name, r.score) for r in second
        ]

    def test_agent_execution_is_repeatable(
        self, demo_scenario: Scenario, config: EvalForgeConfig
    ) -> None:
        first = run_scenario(demo_scenario, config, "candidate")
        second = run_scenario(demo_scenario, config, "candidate")
        assert first.tool_sequence == second.tool_sequence
        assert first.assistant_text() == second.assistant_text()


class TestEvaluatorRegistry:
    """The evaluator set must be complete and correctly wired."""

    def test_all_deterministic_evaluators_are_registered(self) -> None:
        assert len(DETERMINISTIC_EVALUATORS) >= 19

    def test_evaluator_names_are_unique(self, config: EvalForgeConfig) -> None:
        names = [item.name for item in build_evaluators(config)]
        assert len(names) == len(set(names)), f"duplicate evaluator names: {names}"

    def test_a_failing_evaluator_does_not_abort_the_run(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        """One buggy check should cost one dimension, not a 150-session run."""
        from evalforge.evaluators.registry import evaluate_session

        class Exploding:
            name = "exploding"
            kind = "deterministic"

            def evaluate(self, scenario, trace):
                raise RuntimeError("boom")

        evaluators = [*build_evaluators(config), Exploding()]
        results = evaluate_session(demo_scenario, perfect_trace, evaluators)
        assert results, "the surviving evaluators must still produce results"


class TestJudge:
    """Judge reliability controls."""

    def test_mock_judge_is_deterministic(
        self, demo_scenario: Scenario, perfect_trace: SessionTrace
    ) -> None:
        first = MockJudge(seed=42).judge(demo_scenario, perfect_trace)
        second = MockJudge(seed=42).judge(demo_scenario, perfect_trace)
        assert [v.score for v in first.scores] == [v.score for v in second.scores]

    def test_judge_records_its_provenance(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        """A judge score is only interpretable relative to the judge that produced it."""
        results = JudgeEvaluator(config, MockJudge()).evaluate(demo_scenario, perfect_trace)
        assert results
        for item in results:
            assert item.metadata["judge_model"]
            assert item.metadata["judge_prompt_version"]
            assert item.metadata["samples"] == config.evaluation.judge_samples

    def test_judge_aggregates_multiple_samples(
        self, demo_scenario: Scenario, config: EvalForgeConfig, perfect_trace: SessionTrace
    ) -> None:
        results = JudgeEvaluator(config, MockJudge()).evaluate(demo_scenario, perfect_trace)
        assert all(len(item.metadata["sample_scores"]) == 3 for item in results)
        assert any(item.metadata["score_spread"] > 0 for item in results), (
            "samples should vary, otherwise median aggregation is untested"
        )

    def test_judge_requires_evidence(
        self, demo_scenario: Scenario, perfect_trace: SessionTrace
    ) -> None:
        response = MockJudge().judge(demo_scenario, perfect_trace)
        assert all(verdict.evidence for verdict in response.scores)


class TestSemanticFallback:
    """The semantic layer must work with no model available."""

    def test_lexical_similarity_is_bounded_and_symmetric(self) -> None:
        embedder = LexicalEmbedder()
        assert embedder.similarity("launch plan budget", "launch plan budget") > 0.95
        assert 0.0 <= embedder.similarity("apples", "quantum topology") < 0.4

    def test_semantic_threshold_never_fails(self, config: EvalForgeConfig) -> None:
        assert SemanticEvaluator(config).threshold() == 0.0
