"""Trace collection.

The collector is the only writer of :class:`~evalforge.schemas.trace.SessionTrace`. It
owns event ordering and identifier assignment so the agent can emit events without
tracking sequence numbers, and so a trace is guaranteed to be internally consistent
before anything scores it.

Sequence numbers are explicit rather than implied by list position: traces are
serialised, filtered and re-read, and position is not preserved through any of that.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from evalforge.ids import DeterministicIdFactory
from evalforge.schemas.common import EventType, InjectedFailureType, ToolName
from evalforge.schemas.trace import (
    SessionTrace,
    TokenUsage,
    ToolCall,
    TraceEvent,
    TurnRecord,
    WorkspaceSnapshot,
)
from evalforge.tools.base import ToolInvocation


class TraceCollector:
    """Accumulates events and turn records for one session.

    Args:
        run_id: The evaluation run this session belongs to.
        scenario_id: Scenario being executed.
        session_id: Unique identifier for this execution.
        model_name: Model or behaviour profile driving the agent.
        provider_name: Provider serving the session.
        prompt_version: Version of the agent's system prompt.
        agent_version: Version of the agent implementation.
        id_factory: Deterministic identifier source, so a regenerated demo produces
            byte-identical traces.
    """

    def __init__(
        self,
        run_id: str,
        scenario_id: str,
        session_id: str,
        model_name: str = "mock",
        provider_name: str = "mock",
        prompt_version: str = "v1",
        agent_version: str = "v1",
        id_factory: DeterministicIdFactory | None = None,
    ) -> None:
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.session_id = session_id
        self._ids = id_factory or DeterministicIdFactory(seed=session_id)
        self._sequence = 0
        self._events: list[TraceEvent] = []
        self._turns: list[TurnRecord] = []
        self._trace = SessionTrace(
            run_id=run_id,
            scenario_id=scenario_id,
            session_id=session_id,
            model_name=model_name,
            provider_name=provider_name,
            prompt_version=prompt_version,
            agent_version=agent_version,
            started_at=datetime.now(UTC),
        )

    # ------------------------------------------------------------------ events

    def record(
        self,
        event_type: EventType,
        turn_index: int | None = None,
        turn_id: str | None = None,
        **fields: Any,
    ) -> TraceEvent:
        """Append one event and return it.

        Args:
            event_type: What happened.
            turn_index: Turn position, when the event belongs to a turn.
            turn_id: Turn identifier.
            **fields: Any other :class:`TraceEvent` field.
        """
        event = TraceEvent(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            session_id=self.session_id,
            event_id=self._ids.next("event"),
            sequence=self._sequence,
            event_type=event_type,
            turn_index=turn_index,
            turn_id=turn_id,
            **fields,
        )
        self._sequence += 1
        self._events.append(event)
        return event

    def record_tool_invocation(
        self,
        invocation: ToolInvocation,
        turn_index: int,
        turn_id: str,
        retry_count: int = 0,
    ) -> ToolCall:
        """Record a tool call and its outcome as both an event and a ``ToolCall``.

        Both representations are kept: the event log preserves interleaving with model
        calls, while the per-turn ``ToolCall`` list is what tool-use evaluators read.
        """
        if invocation.succeeded:
            event_type = EventType.TOOL_RESULT
        elif not invocation.authorized:
            event_type = EventType.UNAUTHORIZED_ACTION
        else:
            event_type = EventType.TOOL_ERROR

        self.record(
            EventType.TOOL_CALL,
            turn_index=turn_index,
            turn_id=turn_id,
            tool_name=invocation.tool_name,
            tool_arguments=invocation.arguments,
            latency_ms=invocation.latency_ms,
            retry_count=retry_count,
            metadata={"injected_failure": invocation.injected_failure.value},
        )
        self.record(
            event_type,
            turn_index=turn_index,
            turn_id=turn_id,
            tool_name=invocation.tool_name,
            tool_result=invocation.result,
            error=invocation.error,
            latency_ms=invocation.latency_ms,
            retry_count=retry_count,
        )

        return ToolCall(
            tool_call_id=self._ids.next("tool_call"),
            turn_index=turn_index,
            tool_name=invocation.tool_name,
            arguments=invocation.arguments,
            authorized=invocation.authorized,
            result=invocation.result,
            error=invocation.error,
            error_type=invocation.error_type,
            injected_failure=invocation.injected_failure,
            latency_ms=invocation.latency_ms,
            retry_count=retry_count,
            succeeded=invocation.succeeded,
        )

    def record_retry(
        self, tool_name: ToolName, turn_index: int, turn_id: str, attempt: int
    ) -> None:
        """Note that the agent is retrying a failed call."""
        self.record(
            EventType.TOOL_RETRY,
            turn_index=turn_index,
            turn_id=turn_id,
            tool_name=tool_name,
            retry_count=attempt,
        )

    def record_state_transition(
        self, turn_index: int, turn_id: str, from_state: str, to_state: str
    ) -> None:
        """Note an agent state-machine transition."""
        self.record(
            EventType.STATE_TRANSITION,
            turn_index=turn_index,
            turn_id=turn_id,
            metadata={"from": from_state, "to": to_state},
        )

    # ------------------------------------------------------------------- turns

    def add_turn(self, record: TurnRecord) -> None:
        """Append a completed turn record."""
        self._turns.append(record)

    def new_tool_call_id(self) -> str:
        """Mint an identifier for a tool call recorded outside ``record_tool_invocation``."""
        return self._ids.next("tool_call")

    # ------------------------------------------------------------------ finish

    def finish(
        self,
        final_workspace: WorkspaceSnapshot | None = None,
        final_artifact: dict[str, Any] | None = None,
        completed: bool = True,
        abort_reason: str | None = None,
        injected_failures: dict[int, InjectedFailureType] | None = None,
    ) -> SessionTrace:
        """Seal the trace and return it.

        Args:
            final_workspace: Agent state at session end.
            final_artifact: The deliverable produced, if any.
            completed: Whether every turn was executed.
            abort_reason: Why execution stopped early, if it did.
            injected_failures: Turn index to injected fault, recorded on the trace so
                recovery analysis does not have to re-read the scenario.
        """
        self.record(EventType.SESSION_END)
        self._trace.turns = self._turns
        self._trace.events = self._events
        self._trace.final_workspace = final_workspace
        self._trace.final_artifact = final_artifact
        self._trace.completed = completed
        self._trace.abort_reason = abort_reason
        self._trace.completed_at = datetime.now(UTC)
        if injected_failures:
            self._trace.metadata["injected_failures"] = {
                str(index): failure.value for index, failure in injected_failures.items()
            }
        return self._trace

    @property
    def event_count(self) -> int:
        """Events recorded so far."""
        return len(self._events)


def empty_token_usage() -> TokenUsage:
    """A zeroed token counter, for turns where no model call was made."""
    return TokenUsage()
