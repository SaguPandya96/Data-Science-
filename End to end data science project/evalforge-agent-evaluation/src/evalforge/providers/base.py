"""Model provider protocol and request/response contracts.

``ModelProvider`` is a :class:`typing.Protocol` rather than a base class, so a provider
is anything with a ``generate`` method of the right shape. That keeps test doubles and
real SDK wrappers on equal footing and avoids an inheritance hierarchy nobody needs.

The response carries more than text. A model driving an agent has to communicate what it
retained, which constraints it is still tracking, whether it wants approval and whether
it is declining an instruction it found in retrieved content. Making those explicit
fields — rather than parsing them back out of prose — is what lets the trace record the
model's *understanding* separately from its *output*, which is exactly the distinction
context-retention scoring depends on.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from evalforge.schemas.common import ToolName
from evalforge.schemas.trace import TokenUsage


class Message(BaseModel):
    """One entry of conversation history sent to a provider."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(description="user, assistant, system or tool")
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None


class ToolCallRequest(BaseModel):
    """A tool invocation the model wants performed."""

    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Why the model chose this call. Recorded in the trace as evaluator evidence.
    rationale: str = ""


class ModelRequest(BaseModel):
    """Everything a provider needs to produce one response.

    Args:
        system_prompt: The agent's standing instructions.
        messages: Conversation so far, including tool results.
        tools: Function-calling schemas for the available tools.
        temperature: Sampling temperature. Zero for every deterministic run.
        max_tokens: Output cap.
        metadata: Provider-specific context. The mock provider reads simulation state
            from here; external providers ignore it entirely.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    system_prompt: str = ""
    messages: list[Message] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 1024
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """One model turn.

    Args:
        content: Assistant-visible text. Empty when the model is only calling tools.
        tool_calls: Tools the model wants invoked before it replies.
        remembered_facts: What the model believes it should carry forward, as
            ``fact_key -> value``. Context loss shows up here as an absent or reverted
            key, which is a far more precise signal than searching prose for a number.
        active_constraints: Constraint ids the model is still honouring. A dropped id is
            an instruction-forgetting signal.
        request_approval: Tool name the model is asking permission to run, if any.
        refused_injection: Whether the model explicitly declined an instruction it found
            in retrieved content.
        requests_clarification: Whether the model is asking the user to resolve a
            conflict rather than guessing.
        finished: Whether this response ends the turn.
        token_usage: Token accounting, when the provider reports it.
        model_name: Which model or profile produced this.
        raw: Unparsed provider payload, kept for debugging.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    remembered_facts: dict[str, str] = Field(default_factory=dict)
    active_constraints: list[str] = Field(default_factory=list)
    request_approval: str | None = None
    refused_injection: bool = False
    requests_clarification: bool = False
    finished: bool = True
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model_name: str = "unknown"
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        """Whether this response requests tool execution."""
        return bool(self.tool_calls)


@runtime_checkable
class ModelProvider(Protocol):
    """Anything that can turn a :class:`ModelRequest` into a :class:`ModelResponse`."""

    name: str
    model: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Produce one model response."""
        ...

    def available(self) -> bool:
        """Whether this provider can currently serve requests."""
        ...
