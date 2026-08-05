"""Tool infrastructure: typed contracts, invocation, validation and fault injection.

Every productivity tool subclasses :class:`BaseTool` and implements exactly one method,
``_execute``. Everything else — argument validation, approval enforcement, deterministic
latency, fault injection, structured logging, output validation — happens in
:meth:`BaseTool.invoke`, so no individual tool can forget to do it.

All tools are simulated. Nothing leaves the process: no email is sent, no file outside
the run directory is written, no network call is made.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from evalforge.exceptions import (
    ToolError,
    ToolUnauthorizedError,
    ToolValidationError,
)
from evalforge.logging_config import get_logger
from evalforge.rng import jitter
from evalforge.schemas.common import InjectedFailureType, ToolName

logger = get_logger(__name__)

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)


@dataclass
class ToolContext:
    """Everything a tool needs that is not an argument.

    Args:
        run_seed: Master seed, so latency and faults are reproducible.
        scenario_id: Scenario being executed.
        turn_index: Turn during which the call happens.
        call_index: Position of this call within the turn, so two identical calls in one
            turn still get distinct jitter.
        injected_failure: Fault the harness should inject into this call.
        approvals: Actions the user has approved so far this session.
        latency_table: Base latency per tool name, in milliseconds.
        latency_jitter_ratio: Fractional jitter applied to the base latency.
        workspace: Read-only view of agent state, used by tools that operate on the plan.
    """

    run_seed: int
    scenario_id: str
    turn_index: int
    call_index: int = 0
    injected_failure: InjectedFailureType = InjectedFailureType.NONE
    approvals: set[str] = field(default_factory=set)
    latency_table: dict[str, float] = field(default_factory=dict)
    latency_jitter_ratio: float = 0.25
    workspace: dict[str, Any] = field(default_factory=dict)

    def coordinates(self, aspect: str) -> tuple[str | int, ...]:
        """Seed coordinates uniquely identifying one decision in this call."""
        return (self.run_seed, self.scenario_id, self.turn_index, self.call_index, aspect)


@dataclass
class ToolInvocation:
    """The outcome of one tool call, successful or not.

    Kept distinct from the schema-level ``ToolCall`` because this is the tool layer's
    own return value; the tracing layer converts it into the persisted record.
    """

    tool_name: ToolName
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    retryable: bool = False
    authorized: bool = True
    latency_ms: float = 0.0
    injected_failure: InjectedFailureType = InjectedFailureType.NONE

    @property
    def succeeded(self) -> bool:
        """Whether the call produced a result."""
        return self.error is None and self.result is not None


class BaseTool(ABC, Generic[TIn, TOut]):
    """Abstract productivity tool with a typed input and output contract.

    Subclasses declare ``name``, ``input_model``, ``output_model`` and implement
    ``_execute``. They must be pure functions of their arguments and the context: given
    the same inputs they must produce the same output, or the harness loses determinism.
    """

    name: ClassVar[ToolName]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    #: Actions the user must explicitly approve before this tool may run.
    requires_approval: ClassVar[bool] = False
    #: One-line description surfaced to the agent and in the dashboard.
    description: ClassVar[str] = ""

    @abstractmethod
    def _execute(self, payload: TIn, context: ToolContext) -> TOut:
        """Perform the simulated action. Implemented by each concrete tool."""

    def base_latency(self, context: ToolContext) -> float:
        """Deterministic simulated latency for this call, in milliseconds."""
        base = context.latency_table.get(self.name.value, 250.0)
        return jitter(base, context.latency_jitter_ratio, *context.coordinates("latency"))

    def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolInvocation:
        """Validate, execute and post-process one tool call.

        Faults are injected around the real execution rather than inside it, so a tool's
        own logic never contains failure-simulation branches.

        Args:
            arguments: Raw arguments as produced by the model.
            context: Run coordinates, approvals and fault selection.

        Returns:
            A :class:`ToolInvocation` describing the outcome. Errors are returned rather
            than raised, because the agent's recovery behaviour is what is being
            measured and an exception escaping here would abort the session instead.
        """
        from evalforge.tools.failure_injection import (
            apply_post_failure,
            apply_pre_failure,
        )

        latency = self.base_latency(context)
        failure = context.injected_failure
        log = logger.bind(tool=self.name.value, turn=context.turn_index)

        try:
            effective_args = apply_pre_failure(self, arguments, context)

            if self.requires_approval and self.name.value not in context.approvals:
                # The action is still simulated; only the attempt is recorded. Whether
                # the agent should have stopped here is decided by the evaluator.
                raise ToolUnauthorizedError(
                    f"{self.name.value} requires user approval, which was not granted"
                )

            try:
                payload = self.input_model.model_validate(effective_args)
            except ValidationError as exc:
                raise ToolValidationError(
                    f"{self.name.value}: invalid arguments: {_summarise(exc)}"
                ) from exc

            raw = self._execute(payload, context)  # type: ignore[arg-type]
            # The tool's own output is valid by construction: ``_execute`` returns an
            # instance of ``output_model``. Corruption is applied *after* that point, so
            # a fault models a damaged response rather than a buggy tool. This ordering
            # is why a ``MISSING_FIELD`` fault yields a payload the agent must notice,
            # instead of an exception it never sees the contents of.
            result = raw.model_dump(mode="json")
            result = apply_post_failure(self, result, context)

        except ToolError as exc:
            latency *= 3.0 if failure is InjectedFailureType.TIMEOUT else 1.0
            log.warning(
                "tool_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                injected_failure=failure.value,
            )
            return ToolInvocation(
                tool_name=self.name,
                arguments=arguments,
                error=str(exc),
                error_type=type(exc).__name__,
                retryable=exc.retryable,
                authorized=not isinstance(exc, ToolUnauthorizedError),
                latency_ms=latency,
                injected_failure=failure,
            )

        log.debug("tool_call_succeeded", latency_ms=round(latency, 1))
        return ToolInvocation(
            tool_name=self.name,
            arguments=arguments,
            result=result,
            latency_ms=latency,
            injected_failure=failure,
        )

    def schema(self) -> dict[str, Any]:
        """JSON schema of the tool's contract, for provider function-calling payloads."""
        return {
            "name": self.name.value,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "requires_approval": self.requires_approval,
        }


def _summarise(exc: ValidationError) -> str:
    """Render a Pydantic validation error as one readable line."""
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
