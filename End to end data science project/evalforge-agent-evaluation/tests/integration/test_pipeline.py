"""End-to-end pipeline tests, all offline with the mock provider."""

from __future__ import annotations

import pytest

from evalforge.analytics.metrics import compute_metrics
from evalforge.config import EvalForgeConfig
from evalforge.orchestration.comparison import compare_runs
from evalforge.orchestration.pipeline import (
    load_run_result,
    reevaluate_run,
    run_evaluation,
)
from evalforge.reporting.release_readiness import build_report
from evalforge.reporting.render import render_markdown, write_comparison, write_report
from evalforge.scenarios.generator import generate_scenarios
from evalforge.schemas.common import ReleaseDecision
from evalforge.schemas.scenario import Scenario
from evalforge.storage.store import RunStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def suite() -> list[Scenario]:
    """A small but category-complete suite."""
    from evalforge.config import load_config

    return generate_scenarios(count=24, seed=42, config=load_config())


class TestFullRun:
    """Scenarios in, a scored and persisted run out."""

    def test_run_persists_everything(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")

        assert result.summary.session_count == len(suite)
        assert len(result.session_summaries) == len(suite)
        assert store.run_exists(result.run_id)
        assert len(store.get_sessions(result.run_id)) == len(suite)
        assert len(store.get_scenarios(result.run_id)) == len(suite)
        assert len(list(store.iter_traces(result.run_id))) == len(suite)
        assert store.get_evaluations(result.run_id)

    def test_stored_run_reloads_identically(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        """Reading a run back must reproduce its numbers exactly."""
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        reloaded = load_run_result(result.run_id, store)

        assert reloaded.summary.run_id == result.run_id
        assert len(reloaded.session_summaries) == len(result.session_summaries)
        assert reloaded.metrics.get("pass_rate") == pytest.approx(result.metrics.get("pass_rate"))

    def test_run_is_deterministic(
        self, suite: list[Scenario], config: EvalForgeConfig, tmp_path
    ) -> None:
        """The property the regression gate rests on (ADR-003)."""
        first = run_evaluation(
            suite, config, RunStore(tmp_path / "a"), label="x", profile="candidate"
        )
        second = run_evaluation(
            suite, config, RunStore(tmp_path / "b"), label="x", profile="candidate"
        )
        assert first.run_id == second.run_id
        assert [s.overall_score for s in first.session_summaries] == [
            s.overall_score for s in second.session_summaries
        ]
        assert first.critical_failure_count == second.critical_failure_count

    def test_stored_decision_matches_the_report(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        """The run summary's verdict must equal what the release report computes.

        These were two independent code paths, and the stored one was a hardcoded FAIL
        that only became correct if `evalforge report` happened to run afterwards. Any
        reader of the store directly, the dashboard and `evalforge runs`, showed a
        passing agent as failing.
        """
        from evalforge.reporting.release_readiness import build_report

        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        report = build_report(
            result.summary, result.metrics, result.session_summaries, result.results, config
        )

        assert result.summary.release_decision is report.decision
        assert store.get_run(result.run_id).release_decision is report.decision

    def test_a_clean_run_is_not_stored_as_failed(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        """A baseline with no blocking failure must not be recorded as FAIL."""
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        assert result.critical_failure_count == 0
        assert result.summary.release_decision is not ReleaseDecision.FAIL

    def test_reevaluation_needs_no_agent_rerun(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        """Traces are the source of truth, so scoring can be redone later."""
        original = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        rescored = reevaluate_run(original.run_id, config, store)
        assert len(rescored.session_summaries) == len(original.session_summaries)
        assert rescored.metrics.get("overall_score") == pytest.approx(
            original.metrics.get("overall_score")
        )


class TestBaselineVersusCandidate:
    """The regression demonstration."""

    @pytest.fixture(scope="class")
    def runs(self, suite: list[Scenario], config: EvalForgeConfig, tmp_path_factory):
        store = RunStore(tmp_path_factory.mktemp("runs"))
        baseline = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        candidate = run_evaluation(suite, config, store, label="candidate", profile="candidate")
        return baseline, candidate, store

    def test_candidate_scores_worse(self, runs) -> None:
        baseline, candidate, _ = runs
        assert candidate.metrics.get("overall_score") < baseline.metrics.get("overall_score")
        assert candidate.pass_rate < baseline.pass_rate

    def test_candidate_has_more_critical_failures(self, runs) -> None:
        baseline, candidate, _ = runs
        assert candidate.critical_failure_count > baseline.critical_failure_count

    def test_regression_gate_fires(self, runs, config: EvalForgeConfig) -> None:
        baseline, candidate, _ = runs
        report = compare_runs(
            baseline.metrics,
            candidate.metrics,
            config,
            baseline.session_summaries,
            candidate.session_summaries,
        )
        assert report.gate_passed is False
        assert report.regressions

    def test_comparison_reports_an_effect_size(self, runs, config: EvalForgeConfig) -> None:
        baseline, candidate, _ = runs
        report = compare_runs(
            baseline.metrics,
            candidate.metrics,
            config,
            baseline.session_summaries,
            candidate.session_summaries,
        )
        assert "cliffs_delta_overall_score" in report.metadata
        assert report.metadata["cliffs_delta_overall_score"] < 0

    def test_comparison_breaks_down_by_length(self, runs, config: EvalForgeConfig) -> None:
        baseline, candidate, _ = runs
        report = compare_runs(baseline.metrics, candidate.metrics, config)
        assert report.deltas_by_length
        assert report.deltas_by_category

    def test_candidate_fails_release(self, runs, config: EvalForgeConfig) -> None:
        _, candidate, _ = runs
        report = build_report(
            candidate.summary,
            candidate.metrics,
            candidate.session_summaries,
            candidate.results,
            config,
        )
        assert report.decision is ReleaseDecision.FAIL
        assert report.release_blockers


class TestBaselineClearsGates:
    """The reference agent must be shippable, at a sample size where that means something.

    Asserted over a whole suite rather than a single fixture: with small non-zero defect
    rates, any individual conversation is dominated by chance, and the property that
    actually matters is aggregate.
    """

    def test_baseline_records_no_critical_failures(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        assert result.critical_failure_count == 0, (
            "the reference agent must not exhibit a release-blocking failure; "
            f"saw {result.critical_failure_count}"
        )

    def test_baseline_clears_the_blocking_thresholds(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        """Otherwise a candidate failing the gate would be ambiguous: agent or bar?"""
        from evalforge.reporting.release_readiness import evaluate_thresholds

        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        checks = evaluate_thresholds(result.metrics, config)
        failed = [
            check
            for check in checks
            if check.blocking and not check.passed and "NOT MEASURED" not in check.note
        ]
        assert failed == [], (
            "the reference agent fails a blocking gate: "
            f"{[(c.name, round(c.observed, 4), c.threshold) for c in failed]}"
        )


class TestReporting:
    """Report generation and its honesty guarantees."""

    @pytest.fixture(scope="class")
    def report_bundle(self, suite: list[Scenario], config: EvalForgeConfig, tmp_path_factory):
        store = RunStore(tmp_path_factory.mktemp("report_runs"))
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        composition: dict[str, int] = {}
        for scenario in suite:
            composition[scenario.category.value] = composition.get(scenario.category.value, 0) + 1
        report = build_report(
            result.summary,
            result.metrics,
            result.session_summaries,
            result.results,
            config,
            scenario_composition=composition,
        )
        return report, result, store

    def test_markdown_renders(self, report_bundle, config: EvalForgeConfig) -> None:
        report, _, _ = report_bundle
        markdown = render_markdown(report, config)
        assert "# EvalForge Release-Readiness Report" in markdown
        assert "Executive summary" in markdown
        assert "Known limitations" in markdown

    def test_report_labels_simulated_results(self, report_bundle, config: EvalForgeConfig) -> None:
        """Mock results must never read as a measurement of a real model (ADR-003)."""
        report, _, _ = report_bundle
        assert "simulated model behavior" in report.result_label
        assert "simulated model behavior" in render_markdown(report, config)

    def test_report_never_claims_general_safety(
        self, report_bundle, config: EvalForgeConfig
    ) -> None:
        """The report may *disclaim* safety; it must never assert it.

        Matching bare substrings would flag the disclaimer itself ("does not establish
        that the agent is safe"), so the assertions target claim-shaped phrasings.
        """
        import re

        report, _, _ = report_bundle
        markdown = render_markdown(report, config).lower()

        claim_terms = ("is safe", "production ready", "production-ready", "guaranteed")
        negations = ("not", "never", "cannot", "does not", "no ", "without")

        sentences = re.split(r"(?<=[.!?])\s+|\n", markdown)
        for sentence in sentences:
            if not any(term in sentence for term in claim_terms):
                continue
            assert any(negation in sentence for negation in negations), (
                f"report asserts safety without qualification: {sentence.strip()!r}"
            )

        # And the disclaimer must actually be present, not merely implied.
        assert "does not establish" in markdown or "not a general statement" in markdown

    def test_report_always_states_limitations(self, report_bundle) -> None:
        report, _, _ = report_bundle
        assert len(report.limitations) >= 5

    def test_report_records_the_config_digest(self, report_bundle, config: EvalForgeConfig) -> None:
        report, _, _ = report_bundle
        assert report.metadata["config_digest"] == config.digest

    def test_both_formats_are_written(
        self, report_bundle, config: EvalForgeConfig, tmp_path
    ) -> None:
        report, _, _ = report_bundle
        markdown_path, json_path = write_report(report, config, tmp_path)
        assert markdown_path.exists() and markdown_path.stat().st_size > 1000
        assert json_path.exists() and json_path.stat().st_size > 1000

    def test_comparison_report_is_written(
        self, suite: list[Scenario], config: EvalForgeConfig, tmp_path
    ) -> None:
        store = RunStore(tmp_path / "runs")
        baseline = run_evaluation(suite, config, store, label="b", profile="baseline")
        candidate = run_evaluation(suite, config, store, label="c", profile="candidate")
        report = compare_runs(baseline.metrics, candidate.metrics, config)
        markdown_path, json_path = write_comparison(report, tmp_path)
        assert "Regression Comparison" in markdown_path.read_text(encoding="utf-8")
        assert json_path.exists()


class TestMetricsCoverage:
    """Every metric the release report needs must actually be produced."""

    def test_required_metrics_are_present(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        metrics = compute_metrics(result.run_id, result.session_summaries, result.results)

        required = [
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
            "unsupported_claim_rate",
            "goal_drift_rate",
            "contradiction_rate",
            "prompt_injection_resistance",
            "error_propagation_depth",
            "downstream_impact_score",
            "unnecessary_tool_call_rate",
            "average_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "total_tokens",
            "estimated_cost_per_success",
            "critical_failures",
        ]
        missing = [name for name in required if name not in metrics.scalars]
        assert missing == [], f"metrics not computed: {missing}"

    def test_breakdowns_are_populated(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        metrics = result.metrics
        assert metrics.by_category
        assert metrics.by_difficulty
        assert metrics.by_length
        assert metrics.by_model
        assert metrics.by_prompt_version

    def test_intervals_are_computed(
        self, suite: list[Scenario], config: EvalForgeConfig, store: RunStore
    ) -> None:
        result = run_evaluation(suite, config, store, label="baseline", profile="baseline")
        assert "pass_rate" in result.metrics.intervals
        assert result.metrics.intervals["pass_rate"].has_interval
