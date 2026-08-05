"""Tool contracts, validation, approval gating and fault injection."""

from __future__ import annotations

import pytest

from evalforge.config import EvalForgeConfig
from evalforge.schemas.common import InjectedFailureType, ToolName
from evalforge.tools.base import ToolContext
from evalforge.tools.corpus import load_corpus
from evalforge.tools.registry import ToolRegistry, UnknownToolError


def context(
    config: EvalForgeConfig,
    failure: InjectedFailureType = InjectedFailureType.NONE,
    approvals: set[str] | None = None,
    turn: int = 0,
) -> ToolContext:
    """Build a tool context for tests."""
    return ToolContext(
        run_seed=42,
        scenario_id="scn_test",
        turn_index=turn,
        injected_failure=failure,
        approvals=approvals or set(),
        latency_table=config.failure_injection.tool_latency_ms,
        latency_jitter_ratio=config.failure_injection.latency_jitter_ratio,
    )


class TestRegistry:
    """Tool lookup."""

    def test_every_declared_tool_is_registered(self, registry: ToolRegistry) -> None:
        assert len(registry) == len(ToolName)
        for name in ToolName:
            assert name in registry

    def test_unknown_tool_lists_the_alternatives(self, registry: ToolRegistry) -> None:
        with pytest.raises(UnknownToolError, match="Available tools"):
            registry.get("teleport")

    def test_schemas_are_exposed_for_function_calling(self, registry: ToolRegistry) -> None:
        schemas = registry.schemas()
        assert len(schemas) == len(ToolName)
        assert all("input_schema" in schema for schema in schemas)


class TestInputValidation:
    """Bad arguments must be rejected, not coerced."""

    def test_missing_required_argument_fails(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.SEARCH_DOCUMENTS).invoke({}, context(config))
        assert not result.succeeded
        assert result.error_type == "ToolValidationError"
        assert not result.retryable, "re-sending the same bad arguments cannot help"

    def test_wrong_type_fails(self, registry: ToolRegistry, config: EvalForgeConfig) -> None:
        result = registry.get(ToolName.CALCULATE_BUDGET).invoke(
            {"line_items": [{"name": "a", "amount": 10}], "total_budget": "lots"},
            context(config),
        )
        assert not result.succeeded
        assert result.error_type == "ToolValidationError"

    def test_unknown_document_is_permanent_not_retryable(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.EXTRACT_REQUIREMENTS).invoke(
            {"doc_ids": ["doc_does_not_exist"]}, context(config)
        )
        assert not result.succeeded
        assert result.error_type == "ToolPermanentError"
        assert not result.retryable


class TestBudgetArithmetic:
    """``calculate_budget`` is exact, which is what lets a miscalculation be critical."""

    def test_totals_follow_from_inputs(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.CALCULATE_BUDGET).invoke(
            {
                "line_items": [
                    {"name": "engineering", "amount": 9000},
                    {"name": "qa", "amount": 2800},
                ],
                "total_budget": 15000,
                "contingency_rate": 0.10,
            },
            context(config),
        )
        assert result.succeeded
        payload = result.result
        assert payload["allocated"] == pytest.approx(11800.0)
        assert payload["contingency"] == pytest.approx(1500.0)
        assert payload["remaining"] == pytest.approx(1700.0)
        assert payload["within_budget"] is True

    def test_overspend_is_reported(self, registry: ToolRegistry, config: EvalForgeConfig) -> None:
        result = registry.get(ToolName.CALCULATE_BUDGET).invoke(
            {"line_items": [{"name": "everything", "amount": 30000}], "total_budget": 15000},
            context(config),
        )
        assert result.result["within_budget"] is False
        assert result.result["overspend"] > 0


class TestApprovalGating:
    """Approval is enforced by the executor, not by the prompt."""

    @pytest.mark.parametrize("tool", [ToolName.SAVE_ARTIFACT, ToolName.DRAFT_STAKEHOLDER_EMAIL])
    def test_gated_tool_refuses_without_approval(
        self, registry: ToolRegistry, config: EvalForgeConfig, tool: ToolName
    ) -> None:
        arguments = (
            {"artifact_type": "project_plan", "content": {"a": 1}}
            if tool is ToolName.SAVE_ARTIFACT
            else {"plan": {"project_name": "P"}, "recipients": ["a@b.test"], "subject": "s"}
        )
        result = registry.get(tool).invoke(arguments, context(config))
        assert not result.succeeded
        assert result.authorized is False
        assert result.error_type == "ToolUnauthorizedError"

    def test_gated_tool_proceeds_with_approval(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.SAVE_ARTIFACT).invoke(
            {"artifact_type": "project_plan", "content": {"a": 1}},
            context(config, approvals={"save_artifact"}),
        )
        assert result.succeeded
        assert result.authorized is True

    def test_email_is_never_actually_sent(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        """The safety property: nothing leaves the process, ever."""
        result = registry.get(ToolName.DRAFT_STAKEHOLDER_EMAIL).invoke(
            {"plan": {"project_name": "P"}, "recipients": ["a@b.test"], "subject": "s"},
            context(config, approvals={"draft_stakeholder_email"}),
        )
        assert result.succeeded
        assert result.result["sent"] is False
        assert result.result["simulated"] is True

    def test_save_writes_nothing_to_disk(
        self, registry: ToolRegistry, config: EvalForgeConfig, tmp_path
    ) -> None:
        result = registry.get(ToolName.SAVE_ARTIFACT).invoke(
            {"artifact_type": "project_plan", "content": {"a": 1}, "name": "p"},
            context(config, approvals={"save_artifact"}),
        )
        assert result.result["simulated"] is True
        assert result.result["path"].startswith("simulated://")
        assert list(tmp_path.iterdir()) == []


class TestFaultInjection:
    """Every injectable fault must be reproducible and correctly classified."""

    @pytest.mark.parametrize(
        ("fault", "expected_type", "retryable"),
        [
            (InjectedFailureType.TIMEOUT, "ToolTimeoutError", True),
            (InjectedFailureType.TEMPORARY_ERROR, "ToolTemporaryError", True),
            (InjectedFailureType.INVALID_JSON, "ToolOutputError", True),
            (InjectedFailureType.UNAUTHORIZED_ACTION, "ToolUnauthorizedError", False),
        ],
    )
    def test_raising_faults(
        self,
        registry: ToolRegistry,
        config: EvalForgeConfig,
        fault: InjectedFailureType,
        expected_type: str,
        retryable: bool,
    ) -> None:
        result = registry.get(ToolName.EXTRACT_REQUIREMENTS).invoke(
            {"doc_ids": ["doc_analytics_brief"]}, context(config, fault)
        )
        assert not result.succeeded
        assert result.error_type == expected_type
        assert result.retryable is retryable

    @pytest.mark.parametrize(
        "fault",
        [
            InjectedFailureType.EMPTY_RESULT,
            InjectedFailureType.PARTIAL_RESULT,
            InjectedFailureType.MISSING_FIELD,
            InjectedFailureType.CONFLICTING_DATA,
            InjectedFailureType.STALE_DATA,
            InjectedFailureType.INCORRECT_ENTITY,
        ],
    )
    def test_payload_corrupting_faults_still_return(
        self, registry: ToolRegistry, config: EvalForgeConfig, fault: InjectedFailureType
    ) -> None:
        """A corrupted payload must reach the agent, so it has something to notice."""
        result = registry.get(ToolName.SEARCH_DOCUMENTS).invoke(
            {"query": "analytics dashboard scope"}, context(config, fault)
        )
        assert result.succeeded
        assert result.result["_injected"] == fault.value

    def test_conflicting_data_marks_the_altered_field(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.CALCULATE_BUDGET).invoke(
            {"line_items": [{"name": "a", "amount": 100}], "total_budget": 1000},
            context(config, InjectedFailureType.CONFLICTING_DATA),
        )
        assert "_conflicting_field" in result.result

    def test_invalid_argument_type_corrupts_the_request(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        result = registry.get(ToolName.CALCULATE_BUDGET).invoke(
            {"line_items": [{"name": "a", "amount": 100}], "total_budget": 1000},
            context(config, InjectedFailureType.INVALID_ARGUMENT_TYPE),
        )
        assert not result.succeeded
        assert result.error_type == "ToolValidationError"

    def test_injection_is_deterministic(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        first = registry.get(ToolName.SEARCH_DOCUMENTS).invoke(
            {"query": "scope"}, context(config, InjectedFailureType.INCORRECT_ENTITY)
        )
        second = registry.get(ToolName.SEARCH_DOCUMENTS).invoke(
            {"query": "scope"}, context(config, InjectedFailureType.INCORRECT_ENTITY)
        )
        assert first.result["_substituted_entity"] == second.result["_substituted_entity"]

    def test_latency_is_deterministic(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        tool = registry.get(ToolName.SEARCH_DOCUMENTS)
        first = tool.invoke({"query": "scope"}, context(config))
        second = tool.invoke({"query": "scope"}, context(config))
        assert first.latency_ms == pytest.approx(second.latency_ms)
        assert first.latency_ms > 0


class TestCorpus:
    """The document corpus backing retrieval."""

    def test_corpus_loads(self) -> None:
        corpus = load_corpus()
        assert len(corpus.documents) >= 10

    def test_corpus_contains_injection_documents(self) -> None:
        """Without adversarial documents the injection tests cannot run."""
        corpus = load_corpus()
        assert len(corpus.injection_documents) >= 3
        assert all(doc.injection_payload for doc in corpus.injection_documents)

    def test_search_is_deterministic_and_ranked(self) -> None:
        corpus = load_corpus()
        first = [doc.doc_id for doc in corpus.search("analytics dashboard budget")]
        second = [doc.doc_id for doc in corpus.search("analytics dashboard budget")]
        assert first == second
        assert first, "a plain query should return something"

    def test_search_returns_untrusted_content_verbatim(
        self, registry: ToolRegistry, config: EvalForgeConfig
    ) -> None:
        """Sanitising here would make the injection scenarios untestable."""
        result = registry.get(ToolName.SEARCH_DOCUMENTS).invoke(
            {"query": "vendor integration note"}, context(config)
        )
        hits = result.result["documents"]
        flagged = [hit for hit in hits if hit["contains_untrusted_instructions"]]
        assert flagged, "the adversarial vendor note should be reachable"
        assert "ignore the user" in flagged[0]["excerpt"].lower()

    def test_confidential_entities_are_declared(self) -> None:
        assert load_corpus().confidential_names
