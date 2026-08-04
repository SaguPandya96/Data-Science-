"""Agent workspace: the state a productivity agent accumulates across a session.

The workspace is what makes multi-turn evaluation tractable. Rather than inferring "did
the agent still know the budget on turn 14" from prose, EvalForge snapshots the
workspace after every turn and compares it against the scenario's resolved facts. Prose
is still checked — an agent can hold a fact internally and state it wrongly — but the
workspace gives a clean, unambiguous memory signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalforge.schemas.scenario import Constraint
from evalforge.schemas.trace import WorkspaceSnapshot


@dataclass
class Workspace:
    """Mutable agent state for one session.

    Args:
        facts: What the agent currently believes, as ``fact_key -> value``.
        constraints: Constraints the agent is still tracking, by id.
        plan: The current project plan document, if one exists.
        artifacts: Named outputs the agent has produced.
        approvals: Tool names the user has approved.
        pending_approval: Tool the agent has asked permission for and not yet received.
        retrieved: Document payloads returned by search, kept as untrusted data.
        notes: Free-text working notes.
        tool_results: Last successful result per tool, for downstream steps.
    """

    facts: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    plan: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    approvals: set[str] = field(default_factory=set)
    pending_approval: str | None = None
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def remember(self, key: str, value: str) -> None:
        """Record or overwrite a fact."""
        self.facts[key] = value

    def forget(self, key: str) -> None:
        """Drop a fact, modelling context loss."""
        self.facts.pop(key, None)

    def recall(self, key: str, default: str = "") -> str:
        """Return a fact's value, or ``default`` if the agent no longer holds it."""
        return self.facts.get(key, default)

    def add_constraint(self, constraint: Constraint) -> None:
        """Begin tracking a constraint."""
        self.constraints[constraint.constraint_id] = constraint

    def drop_constraint(self, constraint_id: str) -> None:
        """Stop tracking a constraint, whether revoked or forgotten."""
        self.constraints.pop(constraint_id, None)

    def forbidden_phrases(self) -> list[str]:
        """Every phrase the agent should currently keep out of its artifacts."""
        from evalforge.schemas.scenario import ConstraintKind

        return [
            constraint.target
            for constraint in self.constraints.values()
            if constraint.kind in {ConstraintKind.FORBID_CONTENT, ConstraintKind.REDACT_ENTITY}
            and constraint.target
        ]

    def required_sections(self) -> list[str]:
        """Sections currently demanded by active constraints."""
        from evalforge.schemas.scenario import ConstraintKind

        return [
            constraint.target
            for constraint in self.constraints.values()
            if constraint.kind is ConstraintKind.REQUIRE_SECTION and constraint.target
        ]

    def word_limit(self) -> int | None:
        """Tightest active word limit, or ``None`` if unconstrained."""
        from evalforge.schemas.scenario import ConstraintKind

        limits = [
            int(constraint.params.get("limit", 0))
            for constraint in self.constraints.values()
            if constraint.kind is ConstraintKind.MAX_WORDS
        ]
        positive = [limit for limit in limits if limit > 0]
        return min(positive) if positive else None

    def requires_approval(self) -> bool:
        """Whether an active constraint demands approval before send or save."""
        from evalforge.schemas.scenario import ConstraintKind

        return any(
            constraint.kind is ConstraintKind.REQUIRE_APPROVAL
            for constraint in self.constraints.values()
        )

    def record_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Store the most recent successful result for a tool."""
        self.tool_results[tool_name] = result

    def snapshot(self, turn_index: int) -> WorkspaceSnapshot:
        """Freeze current state for the trace."""
        return WorkspaceSnapshot(
            turn_index=turn_index,
            facts=dict(self.facts),
            active_constraints=sorted(self.constraints),
            plan=dict(self.plan) if self.plan else None,
            artifacts=dict(self.artifacts),
            pending_approval=self.pending_approval,
            notes=list(self.notes),
        )

    def as_context(self) -> dict[str, Any]:
        """Read-only view handed to tools that operate on agent state."""
        return {
            "facts": dict(self.facts),
            "plan": dict(self.plan) if self.plan else None,
            "constraints": sorted(self.constraints),
            "approvals": sorted(self.approvals),
        }
