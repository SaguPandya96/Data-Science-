"""Schema validation and serialisation.

The round-trip tests are load-bearing. A model whose ``model_dump_json()`` output its
own validator rejects makes every persisted record unreadable — a defect that does not
surface until something tries to *read* a run, which can be long after it was written.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalforge.schemas.common import (
    CRITICAL_FAILURE_CATEGORIES,
    Dimension,
    FailureCategory,
    Severity,
    dimension_for,
    is_critical,
)
from evalforge.schemas.evaluation import (
    EvaluationResult,
    Evidence,
    ReleaseReadinessReport,
    RunSummary,
    SessionSummary,
)
from evalforge.schemas.scenario import (
    Constraint,
    ConstraintKind,
    ConversationTurn,
    Fact,
    Scenario,
)
from evalforge.schemas.trace import SessionTrace, TokenUsage, ToolCall, TurnRecord


class TestRoundTrips:
    """Every persisted model must survive dump -> load unchanged."""

    def test_session_trace_round_trip(self, perfect_trace: SessionTrace) -> None:
        restored = SessionTrace.model_validate_json(perfect_trace.model_dump_json())
        assert restored.session_id == perfect_trace.session_id
        assert restored.turn_count == perfect_trace.turn_count
        assert len(restored.events) == len(perfect_trace.events)
        assert restored.tool_sequence == perfect_trace.tool_sequence

    def test_token_usage_round_trip(self) -> None:
        """``total_tokens`` must stay a property, not a serialised field.

        A computed field would be written out and then rejected on read by
        ``extra="forbid"``, which is exactly the bug this guards against.
        """
        usage = TokenUsage(input_tokens=120, output_tokens=45)
        assert usage.total_tokens == 165
        restored = TokenUsage.model_validate_json(usage.model_dump_json())
        assert restored.total_tokens == 165

    def test_evaluation_result_round_trip(self) -> None:
        result = EvaluationResult(
            evaluation_id="eva_1",
            run_id="run_1",
            scenario_id="scn_1",
            evaluator_name="fact_retention",
            dimension=Dimension.CONTEXT_RETENTION,
            score=0.0,
            passed=False,
            failure_category=FailureCategory.DATE_LOST,
            severity=Severity.CRITICAL,
            evidence=[Evidence(kind="workspace_fact", excerpt="launch_date missing")],
        )
        assert result.is_critical_failure is True
        restored = EvaluationResult.model_validate_json(result.model_dump_json())
        assert restored.is_critical_failure is True
        assert restored.evidence[0].excerpt == "launch_date missing"

    def test_session_summary_round_trip(self) -> None:
        summary = SessionSummary(
            run_id="run_1",
            session_id="ses_1",
            scenario_id="scn_1",
            overall_score=0.87,
            passed=True,
            failure_categories=["goal_drift"],
        )
        restored = SessionSummary.model_validate_json(summary.model_dump_json())
        assert restored.overall_score == pytest.approx(0.87)
        assert restored.failure_categories == ["goal_drift"]

    def test_run_summary_round_trip(self) -> None:
        summary = RunSummary(run_id="run_1", label="baseline", metrics={"pass_rate": 0.9})
        restored = RunSummary.model_validate_json(summary.model_dump_json())
        assert restored.metrics["pass_rate"] == pytest.approx(0.9)

    def test_release_report_round_trip(self) -> None:
        report = ReleaseReadinessReport(run_id="run_1", executive_summary="ok")
        restored = ReleaseReadinessReport.model_validate_json(report.model_dump_json())
        assert restored.run_id == "run_1"

    def test_scenario_round_trip(self, demo_scenario: Scenario) -> None:
        restored = Scenario.model_validate_json(demo_scenario.model_dump_json())
        assert restored.scenario_id == demo_scenario.scenario_id
        assert restored.turn_count == demo_scenario.turn_count
        assert restored.final_fact_values() == demo_scenario.final_fact_values()


class TestScenarioValidation:
    """Structural invariants a scenario must satisfy."""

    def test_rejects_unknown_constraint_reference(self) -> None:
        with pytest.raises(ValidationError, match="unknown constraint"):
            Scenario(
                scenario_id="s",
                name="n",
                description="d",
                category="goal_drift",
                difficulty="easy",
                conversation_turns=[
                    ConversationTurn(
                        turn_id="t0", turn_index=0, content="hi", constraints_added=["nope"]
                    )
                ],
            )

    def test_rejects_out_of_order_turns(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            Scenario(
                scenario_id="s",
                name="n",
                description="d",
                category="goal_drift",
                difficulty="easy",
                conversation_turns=[
                    ConversationTurn(turn_id="t1", turn_index=1, content="second"),
                    ConversationTurn(turn_id="t0", turn_index=0, content="first"),
                ],
            )

    def test_injected_failure_requires_a_target_tool(self) -> None:
        with pytest.raises(ValidationError, match="requires injected_failure_tool"):
            ConversationTurn(turn_id="t0", turn_index=0, content="go", injected_failure="timeout")

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConversationTurn(turn_id="t", turn_index=0, content="c", nonsense=True)


class TestFactResolution:
    """Later turns must override earlier ones."""

    def test_final_fact_values_uses_latest_value(self, demo_scenario: Scenario) -> None:
        resolved = demo_scenario.final_fact_values()
        assert resolved["budget"].value == "15000", "the turn-2 revision must win"
        assert resolved["launch_date"].value == "2026-09-15", "the date was never revised"

    def test_fact_key_is_slugified(self) -> None:
        fact = Fact(key="  Launch Date ", value="2026-01-01", turn_introduced=0)
        assert fact.key == "launch_date"

    def test_facts_are_immutable(self) -> None:
        fact = Fact(key="budget", value="100", turn_introduced=0)
        with pytest.raises(ValidationError):
            fact.value = "200"


class TestConstraintLifetime:
    """Constraints are active over a half-open turn interval."""

    @pytest.mark.parametrize(
        ("turn", "expected"),
        [(0, False), (2, True), (5, True), (7, False), (9, False)],
    )
    def test_is_active_at(self, turn: int, expected: bool) -> None:
        constraint = Constraint(
            constraint_id="c",
            kind=ConstraintKind.FORBID_CONTENT,
            description="no ads",
            target="ads",
            turn_added=2,
            turn_removed=7,
        )
        assert constraint.is_active_at(turn) is expected

    def test_never_removed_stays_active(self) -> None:
        constraint = Constraint(
            constraint_id="c",
            kind=ConstraintKind.FORBID_CONTENT,
            description="no ads",
            target="ads",
            turn_added=0,
        )
        assert constraint.is_active_at(999) is True


class TestFailureTaxonomy:
    """The taxonomy must stay internally consistent."""

    def test_every_critical_category_maps_to_a_dimension(self) -> None:
        for category in CRITICAL_FAILURE_CATEGORIES:
            assert dimension_for(category) is not None, f"{category} has no dimension"

    def test_every_category_except_none_maps_to_a_dimension(self) -> None:
        unmapped = [
            category
            for category in FailureCategory
            if category is not FailureCategory.NONE and dimension_for(category) is None
        ]
        assert unmapped == [], f"unmapped failure categories: {unmapped}"

    def test_is_critical_matches_the_set(self) -> None:
        assert is_critical(FailureCategory.PROMPT_INJECTION_FOLLOWED) is True
        assert is_critical(FailureCategory.DUPLICATE_TOOL_CALL) is False

    def test_critical_flag_requires_both_severity_and_category(self) -> None:
        """A critical category at low severity must not gate a release."""
        result = EvaluationResult(
            evaluation_id="e",
            run_id="r",
            scenario_id="s",
            evaluator_name="heuristic",
            dimension=Dimension.SAFETY,
            score=0.0,
            passed=False,
            failure_category=FailureCategory.PROMPT_INJECTION_FOLLOWED,
            severity=Severity.MINOR,
        )
        assert result.is_critical_failure is False

    def test_passing_result_is_never_critical(self) -> None:
        result = EvaluationResult(
            evaluation_id="e",
            run_id="r",
            scenario_id="s",
            evaluator_name="x",
            dimension=Dimension.SAFETY,
            score=1.0,
            passed=True,
            failure_category=FailureCategory.PROMPT_INJECTION_FOLLOWED,
            severity=Severity.CRITICAL,
        )
        assert result.is_critical_failure is False


class TestTraceHelpers:
    """Derived views over a trace."""

    def test_totals_aggregate_across_turns(self) -> None:
        trace = SessionTrace(
            run_id="r",
            session_id="s",
            scenario_id="scn",
            turns=[
                TurnRecord(
                    turn_id="t0",
                    turn_index=0,
                    user_message="a",
                    latency_ms=100.0,
                    token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                    tool_calls=[
                        ToolCall(tool_call_id="tc0", turn_index=0, tool_name="search_documents")
                    ],
                ),
                TurnRecord(
                    turn_id="t1",
                    turn_index=1,
                    user_message="b",
                    latency_ms=50.0,
                    token_usage=TokenUsage(input_tokens=20, output_tokens=8),
                ),
            ],
        )
        assert trace.total_latency_ms == pytest.approx(150.0)
        assert trace.total_tokens == 43
        assert trace.turn_count == 2
        assert len(trace.all_tool_calls) == 1
        assert trace.turn_at(1) is not None
        assert trace.turn_at(9) is None
