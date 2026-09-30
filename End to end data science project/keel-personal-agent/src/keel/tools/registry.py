"""Tool plumbing: definitions the model sees, handlers that run, and the risk of each.

Risk levels decide what the agent may do on its own:

- ``read``: looks things up. Always allowed.
- ``write``: changes the user's own local data (a task, a calendar entry, a memory).
  Allowed, and every call is written to the audit log.
- ``outward``: reaches another person or service (sending an email). Never executed by the
  agent. The call becomes a pending approval the user decides on (see ``keel.approvals``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from keel.clock import Clock
from keel.memory.retrieval import KeelMemory
from keel.memory.store import MemoryStore

RISKS = ("read", "write", "outward")


class ToolError(Exception):
    """A problem the model can fix, reported back to it as an error result."""


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    clock: Clock
    memory: MemoryStore
    session_id: str | None = None
    retriever: KeelMemory | None = None  # used by recall; plain Keel (BM25) when unset


Handler = Callable[[ToolContext, dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    risk: str
    handler: Handler

    def definition(self) -> dict[str, Any]:
        """The tool as the Messages API expects it."""
        return {
            "name": self.name,
            "description": f"{self.description} [risk: {self.risk}]",
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.required),
                "additionalProperties": False,
            },
        }


@dataclass
class ToolResult:
    tool_use_id: str
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False
    approval_id: int | None = None


@dataclass
class Toolbox:
    context: ToolContext
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.risk not in RISKS:
            raise ValueError(f"unknown risk {tool.risk!r} for {tool.name}")
        self.tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        # Sorted, so the tool list is byte-identical across requests and stays cached.
        return [self.tools[name].definition() for name in sorted(self.tools)]

    def run(self, tool_use_id: str, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            result = ToolResult(tool_use_id, name, tool_input, f"Unknown tool {name!r}.", True)
        else:
            missing = [key for key in tool.required if key not in tool_input]
            unknown = [key for key in tool_input if key not in tool.properties]
            if missing or unknown:
                problems = []
                if missing:
                    problems.append(f"missing {', '.join(missing)}")
                if unknown:
                    problems.append(f"unexpected {', '.join(unknown)}")
                result = ToolResult(
                    tool_use_id, name, tool_input, f"Invalid input: {'; '.join(problems)}.", True
                )
            elif tool.risk == "outward":
                result = self._queue(tool_use_id, name, tool_input)
            else:
                try:
                    output = tool.handler(self.context, tool_input)
                    result = ToolResult(tool_use_id, name, tool_input, _serialize(output))
                except ToolError as error:
                    result = ToolResult(tool_use_id, name, tool_input, str(error), True)
        self._audit(result)
        return result

    def _queue(self, tool_use_id: str, name: str, tool_input: dict[str, Any]) -> ToolResult:
        from keel.approvals import request_approval

        try:
            approval = request_approval(self.context, name, tool_input)
        except ToolError as error:
            return ToolResult(tool_use_id, name, tool_input, str(error), True)
        message = {
            "status": "pending_approval",
            "approval_id": approval,
            "message": "Queued for the user to approve. Nothing has been sent. "
            "Tell the user it is waiting for their approval.",
        }
        return ToolResult(tool_use_id, name, tool_input, json.dumps(message), approval_id=approval)

    def _audit(self, result: ToolResult) -> None:
        with self.context.conn:
            self.context.conn.execute(
                "INSERT INTO audit (ts, session_id, tool, input, output, is_error)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.context.clock.now().isoformat(),
                    self.context.session_id,
                    result.name,
                    json.dumps(result.input, sort_keys=True),
                    result.output[:4000],
                    int(result.is_error),
                ),
            )


def _serialize(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)


def parse_when(value: Any, field_name: str) -> datetime:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' and explain the format when it's wrong."""
    if not isinstance(value, str):
        raise ToolError(f"{field_name} must be a string like 2026-10-02T14:30")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ToolError(
            f"{field_name}={value!r} is not a date. Use 2026-10-02 or 2026-10-02T14:30."
        ) from None


def require_text(value: Any, field_name: str, limit: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{field_name} must be non-empty text")
    if len(value) > limit:
        raise ToolError(f"{field_name} is longer than {limit} characters")
    return value.strip()
