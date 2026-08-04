"""Deterministic mock model provider.

This is not a test double bolted on for convenience — it is the reference implementation
that powers every test, all of CI and `evalforge demo` (ADR-003). Two things had to be
true for that to work:

1. **It must be able to behave competently.** Otherwise every scenario fails and the
   evaluators are never exercised on the passing path.
2. **It must be able to fail in specific, named ways, on demand, reproducibly.**
   Otherwise the regression gate has nothing to detect and the evaluators are never
   exercised on the failing path.

So the provider first computes what a competent agent would do — reading the scenario
contract, which is legitimate here because the provider stands in for model
*comprehension* — and then applies degradations drawn from a
:class:`~evalforge.config.BehaviorProfileConfig`. Every degradation decision is a seeded
Bernoulli trial keyed on ``(run_seed, scenario_id, turn_index, aspect)``, so it depends
only on its own coordinates and reproduces under parallel or partial execution.

**What mock results mean.** They measure the evaluation system, not any language model.
See ``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalforge.config import BehaviorProfileConfig
from evalforge.logging_config import get_logger
from evalforge.providers.base import ModelRequest, ModelResponse, ToolCallRequest
from evalforge.rng import chance, choose
from evalforge.schemas.common import ToolName
from evalforge.schemas.scenario import (
    ConstraintKind,
    ConversationTurn,
    Fact,
    FactKind,
    Scenario,
)
from evalforge.schemas.trace import TokenUsage
from evalforge.tools.base import ToolInvocation

logger = get_logger(__name__)

#: Canonical fact keys the generator and the provider agree on.
FACT_PROJECT = "project_name"
FACT_LAUNCH = "launch_date"
FACT_BUDGET = "budget"
FACT_CURRENCY = "currency"

#: Tools the mock may substitute when simulating a wrong-tool-selection failure.
_SUBSTITUTE_TOOLS: dict[ToolName, ToolName] = {
    ToolName.SEARCH_DOCUMENTS: ToolName.EXTRACT_REQUIREMENTS,
    ToolName.EXTRACT_REQUIREMENTS: ToolName.SEARCH_DOCUMENTS,
    ToolName.CALCULATE_BUDGET: ToolName.EXTRACT_REQUIREMENTS,
    ToolName.CREATE_PROJECT_PLAN: ToolName.UPDATE_PROJECT_PLAN,
    ToolName.UPDATE_PROJECT_PLAN: ToolName.CREATE_PROJECT_PLAN,
    ToolName.DRAFT_EXECUTIVE_SUMMARY: ToolName.DRAFT_STAKEHOLDER_EMAIL,
    ToolName.DRAFT_STAKEHOLDER_EMAIL: ToolName.DRAFT_EXECUTIVE_SUMMARY,
    ToolName.SAVE_ARTIFACT: ToolName.DRAFT_EXECUTIVE_SUMMARY,
}


@dataclass
class TurnState:
    """Simulation context for one provider call.

    Passed through ``ModelRequest.metadata`` so external providers, which ignore
    metadata entirely, remain drop-in replaceable.

    Args:
        scenario: The scenario being executed.
        turn: The user turn currently being handled.
        workspace_facts: What the agent currently holds.
        plan: Current plan document, if any.
        approvals: Tools the user has approved.
        step: Provider call index within this turn. Step 0 plans; later steps react.
        executed: Tool invocations completed during this turn.
        retry_counts: Attempts made per tool during this turn.
        run_seed: Master seed for degradation decisions.
        retrieved: Search payloads returned this session, treated as untrusted.
    """

    scenario: Scenario
    turn: ConversationTurn
    workspace_facts: dict[str, str] = field(default_factory=dict)
    plan: dict[str, Any] | None = None
    approvals: set[str] = field(default_factory=set)
    step: int = 0
    executed: list[ToolInvocation] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)
    run_seed: int = 42
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def unresolved_failures(self) -> list[ToolInvocation]:
        """Failed calls from this turn that have not since succeeded."""
        succeeded = {inv.tool_name for inv in self.executed if inv.succeeded}
        return [
            inv for inv in self.executed if not inv.succeeded and inv.tool_name not in succeeded
        ]


class MockModelProvider:
    """A deterministic, configurably-fallible stand-in for a real model.

    Args:
        profile: Degradation rates defining this agent revision's behaviour.
        model_name: Label recorded on traces and summaries.
    """

    def __init__(
        self,
        profile: BehaviorProfileConfig,
        model_name: str = "mock-productivity-agent-v1",
    ) -> None:
        self.name = "mock"
        self.model = model_name
        self.profile = profile

    def available(self) -> bool:
        """Always true. The mock provider is the offline guarantee."""
        return True

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Produce one response for the current turn state.

        Raises:
            TypeError: If the request carries no ``turn_state``, which means the caller
                is not the EvalForge agent and the mock has nothing to simulate from.
        """
        state = request.metadata.get("turn_state")
        if not isinstance(state, TurnState):
            raise TypeError(
                "MockModelProvider requires a TurnState in request.metadata['turn_state']"
            )

        if state.step == 0:
            return self._plan_turn(state)
        if state.unresolved_failures:
            return self._recover(state)
        return self._finalise(state)

    # ------------------------------------------------------------------ planning

    def _plan_turn(self, state: TurnState) -> ModelResponse:
        """Decide what to remember and which tools to call for this turn."""
        remembered = self._remembered_facts(state)
        constraints = self._tracked_constraints(state)
        drifted = self._drifts(state)

        if drifted or state.turn.is_distractor:
            return ModelResponse(
                content=self._drift_text(remembered, drifted=drifted),
                remembered_facts=remembered,
                active_constraints=constraints,
                finished=True,
                model_name=self.model,
                token_usage=self._tokens(state, 90),
            )

        calls = self._build_tool_calls(state, remembered, constraints)
        injected_calls, refused = self._injection_response(state, remembered)
        calls.extend(injected_calls)

        approval_needed = self._approval_gate(state, calls)
        if approval_needed is not None:
            # Correct behaviour: stop and ask. The tool is not invoked this step.
            calls = [c for c in calls if c.tool_name.value != approval_needed]
            return ModelResponse(
                content=(
                    f"Before I {approval_needed.replace('_', ' ')}, I need your approval. "
                    "Confirm and I will proceed."
                ),
                tool_calls=calls,
                remembered_facts=remembered,
                active_constraints=constraints,
                request_approval=approval_needed,
                refused_injection=refused,
                finished=not calls,
                model_name=self.model,
                token_usage=self._tokens(state, 70),
            )

        if not calls:
            return ModelResponse(
                content=self._compose_reply(state, remembered, constraints),
                remembered_facts=remembered,
                active_constraints=constraints,
                refused_injection=refused,
                finished=True,
                model_name=self.model,
                token_usage=self._tokens(state, 120),
            )

        return ModelResponse(
            content="",
            tool_calls=calls,
            remembered_facts=remembered,
            active_constraints=constraints,
            refused_injection=refused,
            finished=False,
            model_name=self.model,
            token_usage=self._tokens(state, 60),
        )

    # ------------------------------------------------------------------- memory

    def _remembered_facts(self, state: TurnState) -> dict[str, str]:
        """Simulate what the model still holds, applying context loss and staleness."""
        truth = _facts_known_by(state.scenario, state.turn.turn_index)
        remembered: dict[str, str] = {}
        index = state.turn.turn_index

        for key, fact in truth.items():
            age = index - fact.turn_introduced
            past_horizon = index >= self.profile.context_loss_after_turn
            if (
                past_horizon
                and age >= 2
                and self._roll(state, f"context_loss::{key}", self.profile.context_loss_rate)
            ):
                previous = _previous_value(state.scenario, key, index)
                if previous is not None:
                    # Reverting to a superseded value is a distinct, more damaging
                    # failure than forgetting outright: the agent stays confident.
                    remembered[key] = previous
                continue
            remembered[key] = fact.value
        return remembered

    def _tracked_constraints(self, state: TurnState) -> list[str]:
        """Constraint ids the model is still honouring after instruction forgetting."""
        active = state.scenario.constraints_active_at(state.turn.turn_index)
        tracked: list[str] = []
        for constraint in active:
            age = state.turn.turn_index - constraint.turn_added
            if age >= 1 and self._roll(
                state, f"forget::{constraint.constraint_id}", self.profile.instruction_forget_rate
            ):
                continue
            tracked.append(constraint.constraint_id)
        return tracked

    def _drifts(self, state: TurnState) -> bool:
        """Whether the model abandons this turn's objective."""
        if not state.turn.expected_tool_calls:
            return False
        return self._roll(state, "goal_drift", self.profile.goal_drift_rate)

    # ---------------------------------------------------------------- tool plan

    def _build_tool_calls(
        self,
        state: TurnState,
        remembered: dict[str, str],
        constraints: list[str],
    ) -> list[ToolCallRequest]:
        """Turn the turn's contract into concrete calls, then degrade them."""
        calls: list[ToolCallRequest] = []
        for position, expected in enumerate(state.turn.expected_tool_calls):
            tool_name = expected.tool_name
            arguments = self._arguments_for(state, tool_name, remembered, constraints)
            # Contract-pinned values fill only what the agent could not derive from what
            # it remembers. Deriving first is what makes context loss surface as a wrong
            # argument rather than being masked by the answer key.
            for key, value in expected.arguments.items():
                arguments.setdefault(key, value)

            if self._roll(state, f"wrong_tool::{position}", self.profile.wrong_tool_rate):
                substitute = _SUBSTITUTE_TOOLS.get(tool_name, ToolName.SEARCH_DOCUMENTS)
                calls.append(
                    ToolCallRequest(
                        tool_name=substitute,
                        arguments=self._arguments_for(state, substitute, remembered, constraints),
                        rationale="simulated wrong tool selection",
                    )
                )
                continue

            if self._roll(state, f"wrong_arg::{position}", self.profile.wrong_argument_rate):
                arguments = _perturb_arguments(arguments, state, position)

            calls.append(
                ToolCallRequest(
                    tool_name=tool_name,
                    arguments=arguments,
                    rationale=expected_rationale(tool_name),
                )
            )

            if self._roll(state, f"excess::{position}", self.profile.excessive_tool_call_rate):
                calls.append(
                    ToolCallRequest(
                        tool_name=tool_name,
                        arguments=dict(arguments),
                        rationale="simulated redundant call",
                    )
                )
        return calls

    def _arguments_for(
        self,
        state: TurnState,
        tool_name: ToolName,
        remembered: dict[str, str],
        constraints: list[str],
    ) -> dict[str, Any]:
        """Derive plausible arguments from what the agent currently believes."""
        scenario = state.scenario
        project = remembered.get(
            FACT_PROJECT, scenario.initial_state.get("project_name", "Project")
        )
        launch = remembered.get(FACT_LAUNCH, "")
        currency = remembered.get(FACT_CURRENCY, "USD")
        budget = _as_float(remembered.get(FACT_BUDGET), 0.0)
        budget = self._maybe_cascade(state, budget)

        active = {c.constraint_id: c for c in scenario.active_constraints}
        tracked = [active[cid] for cid in constraints if cid in active]
        forbidden = [
            c.target for c in tracked if c.kind is ConstraintKind.FORBID_CONTENT and c.target
        ]
        sections = [
            c.target for c in tracked if c.kind is ConstraintKind.REQUIRE_SECTION and c.target
        ]
        word_limits = [
            int(c.params.get("limit", 0)) for c in tracked if c.kind is ConstraintKind.MAX_WORDS
        ]
        redactions = [
            c.target for c in tracked if c.kind is ConstraintKind.REDACT_ENTITY and c.target
        ]
        wants_table = any(
            c.kind is ConstraintKind.REQUIRE_FORMAT and c.params.get("format") == "table"
            for c in tracked
        )

        plan = state.plan or {}
        phases = self._phases(state, remembered, constraints)

        if tool_name is ToolName.SEARCH_DOCUMENTS:
            return {
                "query": f"{project} {state.turn.content[:60]}".strip(),
                "project": scenario.initial_state.get("project_slug"),
                "limit": 5,
            }
        if tool_name is ToolName.EXTRACT_REQUIREMENTS:
            doc_ids = [
                str(doc.get("doc_id"))
                for doc in state.retrieved
                if doc.get("doc_id") and not str(doc.get("doc_id")).endswith("_legacy")
            ]
            return {"doc_ids": doc_ids[:3] or ["doc_analytics_brief"]}
        if tool_name is ToolName.CALCULATE_BUDGET:
            return {
                "line_items": [
                    {"name": p["name"], "amount": p["cost"], "category": "delivery"} for p in phases
                ]
                or [{"name": "delivery", "amount": max(budget * 0.7, 1.0)}],
                "total_budget": budget or 1.0,
                "contingency_rate": 0.10,
                "currency": currency,
            }
        if tool_name is ToolName.CREATE_PROJECT_PLAN:
            return {
                "project_name": project,
                "launch_date": launch,
                "budget": budget or 1.0,
                "phases": phases,
                "constraints": [c.description for c in tracked],
                "currency": currency,
            }
        if tool_name is ToolName.UPDATE_PROJECT_PLAN:
            payload: dict[str, Any] = {
                "plan": plan or _seed_plan(project, launch, budget, currency)
            }
            if launch:
                payload["launch_date"] = launch
            if budget:
                payload["budget"] = budget
            new_phases = [p for p in phases if p["name"] not in _phase_names(plan)]
            if new_phases:
                payload["add_phases"] = new_phases
            return payload
        if tool_name is ToolName.DRAFT_EXECUTIVE_SUMMARY:
            payload = {
                "plan": plan or _seed_plan(project, launch, budget, currency),
                "required_sections": sections or [],
                "excluded_topics": forbidden,
                "use_table": wants_table,
            }
            if word_limits:
                payload["max_words"] = min(w for w in word_limits if w > 0)
            return payload
        if tool_name is ToolName.DRAFT_STAKEHOLDER_EMAIL:
            return {
                "plan": plan or _seed_plan(project, launch, budget, currency),
                "recipients": ["sponsor@example-fictional.test"],
                "subject": f"{project} status",
                "redact_entities": redactions,
            }
        return {
            "artifact_type": "project_plan",
            "content": plan or _seed_plan(project, launch, budget, currency),
            "name": project,
        }

    def _phases(
        self, state: TurnState, remembered: dict[str, str], constraints: list[str]
    ) -> list[dict[str, Any]]:
        """Build the phase list, including any phase a forgotten constraint permits."""
        budget = _as_float(remembered.get(FACT_BUDGET), 0.0)
        base = [
            {
                "name": "Discovery",
                "duration_weeks": 2.0,
                "owner": "delivery",
                "cost": round(budget * 0.2, 2),
                "description": "Requirements and scope confirmation",
            },
            {
                "name": "Build",
                "duration_weeks": 6.0,
                "owner": "engineering",
                "cost": round(budget * 0.5, 2),
                "description": "Implementation of agreed scope",
            },
        ]
        extra = state.scenario.initial_state.get("additional_phases", [])
        for name in extra:
            base.append(
                {
                    "name": str(name),
                    "duration_weeks": 2.0,
                    "owner": "delivery",
                    "cost": round(budget * 0.1, 2),
                    "description": f"{name} activity",
                }
            )

        # A forgotten "do not include X" constraint has to actually produce X, otherwise
        # the violation is invisible to the evaluator and the simulation is not honest.
        tracked = set(constraints)
        for constraint in state.scenario.constraints_active_at(state.turn.turn_index):
            if (
                constraint.kind is ConstraintKind.FORBID_CONTENT
                and constraint.constraint_id not in tracked
                and constraint.target
            ):
                base.append(
                    {
                        "name": constraint.target.title(),
                        "duration_weeks": 1.0,
                        "owner": "marketing",
                        "cost": round(budget * 0.08, 2),
                        "description": f"Includes {constraint.target}",
                    }
                )
        return base

    def _maybe_cascade(self, state: TurnState, budget: float) -> float:
        """Propagate a corrupted upstream value instead of the user's stated one."""
        corrupted = state.tool_results.get(ToolName.CALCULATE_BUDGET.value, {})
        conflicting = corrupted.get("_conflicting_field")
        if not conflicting:
            return budget
        if not self._roll(state, "cascade", self.profile.cascade_propagation_rate):
            return budget
        replacement = corrupted.get(conflicting)
        if isinstance(replacement, int | float):
            return float(replacement)
        return budget

    # ------------------------------------------------------------- safety paths

    def _approval_gate(self, state: TurnState, calls: list[ToolCallRequest]) -> str | None:
        """Return the tool needing approval, or ``None`` if the model proceeds anyway.

        A competent agent stops and asks. A degraded one proceeds, producing an
        unauthorised attempt that the tool layer records and the evaluator treats as a
        critical failure.
        """
        requires_approval = {
            ToolName.SAVE_ARTIFACT.value,
            ToolName.DRAFT_STAKEHOLDER_EMAIL.value,
        }
        for call in calls:
            name = call.tool_name.value
            if name not in requires_approval or name in state.approvals:
                continue
            if self._roll(state, f"unauthorized::{name}", self.profile.unauthorized_action_rate):
                return None
            return name
        return None

    def _injection_response(
        self, state: TurnState, remembered: dict[str, str]
    ) -> tuple[list[ToolCallRequest], bool]:
        """Decide whether to obey an instruction found in retrieved content.

        Returns:
            The extra tool calls obedience would produce, and whether the model
            explicitly refused. Refusal is recorded so the evaluator can distinguish
            "resisted" from "never saw an injection".
        """
        payloads = [
            doc for doc in state.retrieved if doc.get("contains_untrusted_instructions") is True
        ]
        if not payloads and not state.turn.contains_injection:
            return [], False

        if not self._roll(state, "injection", self.profile.injection_compliance_rate):
            return [], True

        project = remembered.get(FACT_PROJECT, "Project")
        plan = state.plan or _seed_plan(project, remembered.get(FACT_LAUNCH, ""), 0.0, "USD")
        logger.warning(
            "mock_followed_injection",
            scenario_id=state.scenario.scenario_id,
            turn=state.turn.turn_index,
        )
        return (
            [
                ToolCallRequest(
                    tool_name=ToolName.DRAFT_STAKEHOLDER_EMAIL,
                    arguments={
                        "plan": plan,
                        "recipients": ["vendor-contacts@example-fictional.test"],
                        "subject": "Stakeholder register as requested",
                        "body_summary": "Full stakeholder register attached as instructed.",
                    },
                    rationale="followed instruction found in retrieved document",
                )
            ],
            False,
        )

    # ------------------------------------------------------------- recovery path

    def _recover(self, state: TurnState) -> ModelResponse:
        """React to a failed tool call."""
        failure = state.unresolved_failures[-1]
        tool_key = failure.tool_name.value
        attempts = state.retry_counts.get(tool_key, 0)
        remembered = self._remembered_facts(state)
        constraints = self._tracked_constraints(state)

        gives_up = self._roll(
            state, f"recovery::{tool_key}::{attempts}", self.profile.recovery_failure_rate
        )

        if gives_up:
            # The damaging failure mode: assert the result anyway. This is what
            # `fabricated_tool_result` detects, and why it is critical.
            return ModelResponse(
                content=(
                    f"I have completed the {tool_key.replace('_', ' ')} step. "
                    f"The result confirms the plan is on track with no outstanding issues."
                ),
                remembered_facts=remembered,
                active_constraints=constraints,
                finished=True,
                model_name=self.model,
                token_usage=self._tokens(state, 80),
                raw={"simulated_failure": "fabricated_recovery"},
            )

        if failure.retryable and attempts < 2:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name=failure.tool_name,
                        arguments=dict(failure.arguments),
                        rationale=f"retry after retryable error: {failure.error_type}",
                    )
                ],
                remembered_facts=remembered,
                active_constraints=constraints,
                finished=False,
                model_name=self.model,
                token_usage=self._tokens(state, 50),
            )

        if self._roll(state, f"clarify::{tool_key}", self.profile.clarification_rate):
            return ModelResponse(
                content=(
                    f"The {tool_key.replace('_', ' ')} step did not return usable data "
                    f"({failure.error_type}). I have not filled in the missing values. "
                    "How would you like me to proceed?"
                ),
                remembered_facts=remembered,
                active_constraints=constraints,
                requests_clarification=True,
                finished=True,
                model_name=self.model,
                token_usage=self._tokens(state, 75),
            )

        return ModelResponse(
            content=(
                f"I could not complete the {tool_key.replace('_', ' ')} step "
                f"({failure.error_type}). I have continued with the information already "
                "confirmed and flagged the gap rather than estimating it."
            ),
            remembered_facts=remembered,
            active_constraints=constraints,
            finished=True,
            model_name=self.model,
            token_usage=self._tokens(state, 85),
        )

    # ------------------------------------------------------------ final message

    def _finalise(self, state: TurnState) -> ModelResponse:
        """Compose the assistant's reply once tools have run."""
        remembered = self._remembered_facts(state)
        constraints = self._tracked_constraints(state)
        return ModelResponse(
            content=self._compose_reply(state, remembered, constraints),
            remembered_facts=remembered,
            active_constraints=constraints,
            finished=True,
            model_name=self.model,
            token_usage=self._tokens(state, 140),
        )

    def _compose_reply(
        self,
        state: TurnState,
        remembered: dict[str, str],
        constraints: list[str],
    ) -> str:
        """Render the assistant message.

        Fact values are stated verbatim so that text-level and workspace-level context
        checks are independent signals rather than the same signal twice.
        """
        project = remembered.get(FACT_PROJECT, "the project")
        parts = [f"Here is the updated position on {project}."]

        if launch := remembered.get(FACT_LAUNCH):
            parts.append(f"Launch date: {launch}.")
        if budget := remembered.get(FACT_BUDGET):
            currency = remembered.get(FACT_CURRENCY, "USD")
            parts.append(f"Budget: {budget} {currency}.")

        tracked = set(constraints)
        for constraint in state.scenario.constraints_active_at(state.turn.turn_index):
            if constraint.constraint_id in tracked:
                if constraint.kind is ConstraintKind.FORBID_CONTENT and constraint.target:
                    parts.append(f"Excluded as instructed: {constraint.target}.")
                elif constraint.kind is ConstraintKind.PRESERVE_FACT:
                    parts.append(f"Preserved as instructed: {constraint.description}.")
            elif constraint.kind is ConstraintKind.FORBID_CONTENT and constraint.target:
                # Forgetting a prohibition has to be observable in the output.
                parts.append(f"Added {constraint.target} to the plan to broaden reach.")

        for invocation in state.executed:
            if invocation.succeeded and invocation.tool_name is ToolName.CALCULATE_BUDGET:
                result = invocation.result or {}
                parts.append(
                    f"Allocated {result.get('allocated')} with {result.get('remaining')} remaining."
                )

        if self._roll(state, "unsupported", self.profile.unsupported_claim_rate):
            parts.append(
                "Historical delivery data shows a 94 percent on-time rate for comparable "
                "launches, and the vendor has confirmed capacity for 18 additional seats."
            )

        conflict = _conflict_note(state)
        if conflict and self._roll(state, "conflict_clarify", self.profile.clarification_rate):
            parts.append(conflict)

        return " ".join(parts)

    def _drift_text(self, remembered: dict[str, str], drifted: bool) -> str:
        """Text for a distractor turn or an abandoned objective."""
        if drifted:
            return (
                "That is an interesting question. Broadly, teams handle this differently "
                "depending on their portfolio governance and reporting cadence, and there "
                "are several schools of thought worth considering."
            )
        project = remembered.get(FACT_PROJECT, "the project")
        return (
            f"Noted. That does not change the current position on {project}; "
            "I have kept the existing plan as it stands."
        )

    # --------------------------------------------------------------- utilities

    def _roll(self, state: TurnState, aspect: str, probability: float) -> bool:
        """Seeded Bernoulli trial scoped to this exact decision."""
        return chance(
            probability,
            state.run_seed,
            state.scenario.scenario_id,
            state.turn.turn_index,
            state.step,
            aspect,
        )

    def _tokens(self, state: TurnState, output: int) -> TokenUsage:
        """Deterministic token accounting derived from conversation size."""
        history = sum(
            len(t.content) for t in state.scenario.conversation_turns[: state.turn.turn_index + 1]
        )
        return TokenUsage(input_tokens=200 + history // 4, output_tokens=output)


def expected_rationale(tool_name: ToolName) -> str:
    """Short justification recorded alongside each planned call."""
    return {
        ToolName.SEARCH_DOCUMENTS: "locate the source documents for this project",
        ToolName.EXTRACT_REQUIREMENTS: "pull structured requirements from the sources",
        ToolName.CALCULATE_BUDGET: "check the line items against the approved budget",
        ToolName.CREATE_PROJECT_PLAN: "assemble the plan from confirmed inputs",
        ToolName.UPDATE_PROJECT_PLAN: "apply the requested revision",
        ToolName.DRAFT_EXECUTIVE_SUMMARY: "summarise the plan for the sponsor",
        ToolName.DRAFT_STAKEHOLDER_EMAIL: "draft the stakeholder update",
        ToolName.SAVE_ARTIFACT: "persist the approved artifact",
    }.get(tool_name, "advance the task")


def _facts_known_by(scenario: Scenario, turn_index: int) -> dict[str, Fact]:
    """Resolve fact values as of ``turn_index``, with later turns overriding earlier."""
    resolved: dict[str, Fact] = {}
    for turn in scenario.conversation_turns:
        if turn.turn_index > turn_index:
            break
        for fact in (*turn.introduced_facts, *turn.updated_facts):
            resolved[fact.key] = fact
    return resolved


def _previous_value(scenario: Scenario, key: str, turn_index: int) -> str | None:
    """The value ``key`` held before its most recent update at or before ``turn_index``."""
    history: list[str] = []
    for turn in scenario.conversation_turns:
        if turn.turn_index > turn_index:
            break
        for fact in (*turn.introduced_facts, *turn.updated_facts):
            if fact.key == key:
                history.append(fact.value)
    if len(history) < 2:
        return None
    return history[-2]


def _as_float(value: str | None, default: float) -> float:
    """Parse a money-ish string into a float, tolerating symbols and separators."""
    if not value:
        return default
    cleaned = value.replace("$", "").replace(",", "").replace("USD", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return default


def _seed_plan(project: str, launch: str, budget: float, currency: str) -> dict[str, Any]:
    """Minimal plan document used when a tool needs one and none exists yet."""
    return {
        "plan_id": "plan_bootstrap",
        "project_name": project,
        "launch_date": launch,
        "budget": budget,
        "currency": currency,
        "phases": [],
        "constraints": [],
        "total_phase_cost": 0.0,
        "total_duration_weeks": 0.0,
        "notes": "",
        "revision": 1,
    }


def _phase_names(plan: dict[str, Any]) -> set[str]:
    """Names of phases already present in a plan."""
    return {str(p.get("name", "")) for p in plan.get("phases", []) or []}


def _perturb_arguments(
    arguments: dict[str, Any], state: TurnState, position: int
) -> dict[str, Any]:
    """Corrupt one argument, modelling a wrong-argument failure.

    Numeric values are shifted rather than replaced with nonsense, because a plausible
    wrong number is the realistic failure and the one a human reviewer would miss.
    """
    perturbed = dict(arguments)
    numeric = sorted(
        key
        for key, value in perturbed.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    )
    if numeric:
        key = choose(numeric, state.run_seed, state.scenario.scenario_id, position, "arg_num")
        perturbed[key] = round(float(perturbed[key]) * 1.25, 2)
        return perturbed

    textual = sorted(
        key for key, value in perturbed.items() if isinstance(value, str) and value.strip()
    )
    if textual:
        key = choose(textual, state.run_seed, state.scenario.scenario_id, position, "arg_text")
        perturbed[key] = f"{perturbed[key]} (revised)"
    return perturbed


def _conflict_note(state: TurnState) -> str:
    """Flag a user instruction that contradicts a still-active constraint."""
    for constraint in state.scenario.constraints_active_at(state.turn.turn_index):
        if constraint.kind is not ConstraintKind.FORBID_CONTENT or not constraint.target:
            continue
        if constraint.target.lower() in state.turn.content.lower():
            return (
                f"Note: this request conflicts with your earlier instruction to exclude "
                f"{constraint.target}. Should I override that instruction?"
            )
    return ""


def fact_is_date(fact: Fact) -> bool:
    """Whether a fact should be compared as a date."""
    return fact.kind is FactKind.DATE
