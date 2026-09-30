"""The language model behind the agent.

``ClaudeModel`` calls the Claude API. ``ScriptedModel`` replays fixed replies, so the whole
agent loop runs in tests and CI without a key or any cost.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class ModelReply:
    content: list[dict[str, Any]]  # the assistant blocks, appended to history unchanged
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    refusal_category: str | None = None


class Model(Protocol):
    def reply(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply: ...


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "to_dict"):
        return dict(block.to_dict())
    return dict(block.model_dump(exclude_none=True))


class ClaudeModel:
    """Claude through the Anthropic SDK.

    - Adaptive thinking is always on for this model; ``effort`` sets how hard it thinks.
    - The system prompt and tool list never change during a session, so they are cached.
      Everything that changes per turn (time, memories, agenda) goes in the user message.
    - Server-side fallbacks are on: if a request is declined by a safety classifier, the
      API retries it on a fallback model inside the same call.
    - Web search runs on Anthropic's side, so results arrive already inside the reply.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        web_search: bool = True,
        max_tokens: int = 16000,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.web_search = web_search
        self.max_tokens = max_tokens

    def reply(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply:
        all_tools = list(tools)
        if self.web_search:
            all_tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": 5})
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=all_tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = response.usage
        category = None
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            category = response.stop_details.category
        return ModelReply(
            content=[_block_to_dict(b) for b in response.content],
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            },
            refusal_category=category,
        )


class ScriptedModel:
    """Replays replies in order and records every request it was sent."""

    def __init__(self, replies: Iterable[ModelReply]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def reply(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply:
        # Deep enough copy that later appends to the history don't rewrite what was sent.
        self.requests.append(
            {"system": system, "messages": [dict(m) for m in messages], "tools": tools}
        )
        if not self.replies:
            raise RuntimeError("ScriptedModel ran out of replies")
        return self.replies.pop(0)


def text(value: str) -> ModelReply:
    """A scripted final answer."""
    return ModelReply(content=[{"type": "text", "text": value}], stop_reason="end_turn")


def tool_calls(*calls: tuple[str, str, dict[str, Any]], preamble: str = "") -> ModelReply:
    """A scripted reply that calls tools: (id, name, input) tuples."""
    content: list[dict[str, Any]] = []
    if preamble:
        content.append({"type": "text", "text": preamble})
    content += [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]
    return ModelReply(content=content, stop_reason="tool_use")


def api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
