"""Trace schema: the complete, replayable record of one agent session.

Everything EvalForge concludes about an agent is derived from a ``SessionTrace``. That
constraint is deliberate — it means an evaluation can be re-run months later against a
stored trace without re-invoking a model, and it means a human annotator and an
automated evaluator are looking at exactly the same evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalforge.schemas.common import EventType, InjectedFailureType, ToolName


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TokenUsage(BaseModel):
    """Token accounting for one model call, when the provider reports it."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    # A plain property, not a computed_field: a computed field would be serialised and
    # then rejected on read by extra="forbid", making stored traces unloadable.
    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class ToolCall(BaseModel):
    """A single tool invocation, including whether it was authorised and what it returned.

    Args:
        tool_call_id: Stable identifier.
        turn_index: Turn during which the call was made.
        tool_name: Which tool was invoked.
        arguments: Arguments the model supplied.
        expected: Whether the scenario contract anticipated this call. Populated by the
            evaluation layer, not the agent, so the trace stays a neutral record.
        authorized: Whether the agent held the approval this action required. A false
            value on a send or save is a critical safety failure.
        result: Structured tool output, if the call succeeded.
        error: Error message, if it failed.
        error_type: Exception class name, for grouping.
        injected_failure: Which fault the harness injected, if any.
        latency_ms: Simulated wall-clock cost.
        retry_count: How many retries preceded the recorded outcome.
        succeeded: Whether the final attempt produced a result.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    turn_index: int = Field(ge=0)
    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected: bool | None = None
    authorized: bool = True
    result: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    injected_failure: InjectedFailureType = InjectedFailureType.NONE
    latency_ms: float = Field(default=0.0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    succeeded: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """One timestamped event in a session.

    Args:
        run_id: The evaluation run this session belongs to.
        scenario_id: Which scenario produced the session.
        session_id: Unique per agent execution of a scenario.
        turn_id: Turn the event occurred within, if applicable.
        turn_index: Numeric turn position, for ordering and length analysis.
        event_id: Stable identifier.
        sequence: Monotonic ordering key within the session.
        timestamp: When the event was recorded.
        event_type: What kind of event this is.
        model_input: Prompt or message list sent to the model, when relevant.
        model_output: Raw model output, when relevant.
        tool_name: Tool involved, when relevant.
        tool_arguments: Arguments passed to the tool.
        tool_result: Structured tool output.
        latency_ms: Cost of this event.
        token_usage: Tokens consumed, when the provider reports them.
        estimated_cost: Estimated USD cost, when a price table is known.
        error: Error text, when the event records a failure.
        retry_count: Retry depth at the time of the event.
        metadata: Anything evaluator-relevant that has no dedicated field.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    session_id: str
    event_id: str
    sequence: int = Field(ge=0)
    event_type: EventType
    turn_id: str | None = None
    turn_index: int | None = None
    timestamp: datetime = Field(default_factory=_utc_now)
    model_input: str | None = None
    model_output: str | None = None
    tool_name: ToolName | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    token_usage: TokenUsage | None = None
    estimated_cost: float = Field(default=0.0, ge=0)
    error: str | None = None
    retry_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSnapshot(BaseModel):
    """The agent's internal state at the end of a turn.

    Context-retention scoring compares this against the scenario's resolved facts. A
    snapshot per turn is what makes it possible to say *when* a fact was dropped rather
    than only that the final answer was wrong.
    """

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    facts: dict[str, str] = Field(default_factory=dict)
    active_constraints: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    pending_approval: str | None = None
    notes: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    """Everything that happened while handling one user turn.

    Args:
        turn_id: Matches the scenario turn.
        turn_index: Position in the conversation.
        user_message: What the user said.
        assistant_message: What the agent replied.
        tool_calls: Tool activity for this turn, in order.
        workspace_after: Agent state once the turn completed.
        latency_ms: Total simulated cost of the turn.
        token_usage: Tokens consumed during the turn.
        clarification_requested: Whether the agent asked the user to resolve a conflict.
        refused_injection: Whether the agent explicitly declined injected instructions.
        errors: Error strings surfaced while handling the turn.
    """

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    turn_index: int = Field(ge=0)
    user_message: str
    assistant_message: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    workspace_after: WorkspaceSnapshot | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    clarification_requested: bool = False
    refused_injection: bool = False
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionTrace(BaseModel):
    """The full record of one agent execution of one scenario.

    Args:
        run_id: Evaluation run identifier.
        session_id: Unique per execution.
        scenario_id: Scenario that was executed.
        model_name: Model or mock profile that drove the agent.
        prompt_version: Version of the agent's system prompt.
        agent_version: Version of the agent implementation.
        provider_name: Which provider served the session.
        started_at: Session start time.
        completed_at: Session end time.
        turns: Per-turn records.
        events: Flat event log, ordered by ``sequence``.
        final_workspace: Agent state at session end.
        final_artifact: The deliverable the agent produced, if any.
        completed: Whether the session ran to the last turn without aborting.
        abort_reason: Why it stopped early, if it did.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    session_id: str
    scenario_id: str
    model_name: str = "mock"
    prompt_version: str = "v1"
    agent_version: str = "v1"
    provider_name: str = "mock"
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    final_workspace: WorkspaceSnapshot | None = None
    final_artifact: dict[str, Any] | None = None
    completed: bool = True
    abort_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        """Every tool call across every turn, in execution order."""
        return [call for turn in self.turns for call in turn.tool_calls]

    @property
    def tool_sequence(self) -> list[ToolName]:
        """The ordered sequence of tools the agent actually invoked."""
        return [call.tool_name for call in self.all_tool_calls]

    @property
    def total_latency_ms(self) -> float:
        """Total simulated latency of the session."""
        return sum(turn.latency_ms for turn in self.turns)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all turns."""
        return sum(turn.token_usage.total_tokens for turn in self.turns)

    @property
    def estimated_cost(self) -> float:
        """Total estimated USD cost recorded on the event log."""
        return sum(event.estimated_cost for event in self.events)

    @property
    def turn_count(self) -> int:
        """Number of turns executed."""
        return len(self.turns)

    def turn_at(self, turn_index: int) -> TurnRecord | None:
        """Return the record for ``turn_index``, or ``None`` if absent."""
        for turn in self.turns:
            if turn.turn_index == turn_index:
                return turn
        return None

    def events_of_type(self, event_type: EventType) -> list[TraceEvent]:
        """Every event matching ``event_type``."""
        return [event for event in self.events if event.event_type is event_type]

    def assistant_text(self) -> str:
        """All assistant output concatenated, for whole-session text checks."""
        return "\n".join(turn.assistant_message for turn in self.turns)
