"""Scenario schema: the specification of one adversarial multi-turn session.

A scenario is a *contract*, not a transcript. It declares what the user says, what the
agent is supposed to remember, which constraints stay live, which tools should fire and
in what order, and what would count as failure. Evaluators read the contract and the
recorded trace and compare them; nothing about correctness is decided at run time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalforge.schemas.common import (
    Difficulty,
    InjectedFailureType,
    ScenarioCategory,
    ToolName,
    TurnRole,
)


class FactKind(StrEnum):
    """Type of a tracked fact, which selects how it is compared.

    Dates and money need exact comparison because a "close enough" deadline is still a
    missed deadline; free text is compared by normalised containment.
    """

    DATE = "date"
    MONEY = "money"
    NUMBER = "number"
    TEXT = "text"
    ENTITY = "entity"
    LIST = "list"


class ConstraintKind(StrEnum):
    """Category of a persistent user instruction.

    Each kind maps to a deterministic check. A constraint EvalForge cannot check
    deterministically does not belong in a scenario contract.
    """

    FORBID_CONTENT = "forbid_content"
    REQUIRE_SECTION = "require_section"
    PRESERVE_FACT = "preserve_fact"
    MAX_WORDS = "max_words"
    REQUIRE_FORMAT = "require_format"
    REQUIRE_APPROVAL = "require_approval"
    SOURCE_RESTRICTION = "source_restriction"
    REDACT_ENTITY = "redact_entity"


class Fact(BaseModel):
    """A single piece of state the user supplied and the agent must retain.

    Args:
        key: Stable slug used to look the fact up, e.g. ``launch_date``.
        value: The canonical value the agent is expected to carry forward.
        kind: How the value should be compared.
        turn_introduced: Index of the turn that first stated the fact.
        critical: If true, losing this fact is a release-blocking failure.
        aliases: Alternative surface forms that count as a correct restatement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    kind: FactKind = FactKind.TEXT
    turn_introduced: int = Field(ge=0)
    critical: bool = False
    aliases: tuple[str, ...] = ()

    @field_validator("key")
    @classmethod
    def _slug_key(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class Constraint(BaseModel):
    """A persistent instruction that stays in force until explicitly revoked.

    Constraints are the reason single-turn evaluation is insufficient: "do not include
    paid advertising" is stated once and must hold for every subsequent artifact.

    Args:
        constraint_id: Stable identifier, unique within the scenario.
        kind: Which deterministic check applies.
        description: The instruction in the user's own words.
        target: The subject of the constraint (a phrase, section name or fact key).
        turn_added: Turn index at which the constraint became active.
        turn_removed: Turn index at which the user revoked it, if ever.
        params: Kind-specific parameters, e.g. ``{"limit": 150}`` for ``MAX_WORDS``.
        critical: If true, violating it blocks release.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint_id: str = Field(min_length=1)
    kind: ConstraintKind
    description: str = Field(min_length=1)
    target: str = ""
    turn_added: int = Field(default=0, ge=0)
    turn_removed: int | None = Field(default=None, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False

    def is_active_at(self, turn_index: int) -> bool:
        """Return whether the constraint is in force at ``turn_index``."""
        if turn_index < self.turn_added:
            return False
        return not (self.turn_removed is not None and turn_index >= self.turn_removed)


class ExpectedToolCall(BaseModel):
    """A tool invocation the scenario expects at a given turn.

    Args:
        tool_name: Which tool should fire.
        arguments: Argument subset that must match. Only the keys present here are
            compared, so a scenario can pin the launch date without pinning every field.
        required: If false, calling the tool is acceptable but not mandatory.
        order_index: Position within the session's expected sequence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    order_index: int = Field(default=0, ge=0)


class ConversationTurn(BaseModel):
    """One turn of the scripted conversation, plus its evaluation contract.

    Args:
        turn_id: Stable identifier for the turn.
        turn_index: Zero-based position in the conversation.
        role: Who is speaking. Scenarios script user and system turns only; assistant
            content is produced by the agent under test.
        content: The literal message text.
        expected_behavior: Prose description of the correct agent response, used by the
            LLM judge and shown to human annotators.
        introduced_facts: Facts first stated in this turn.
        updated_facts: Facts whose value this turn overrides.
        constraints_added: Constraint ids that become active at this turn.
        constraints_removed: Constraint ids the user explicitly revokes here.
        expected_tool_calls: Tool calls the agent should make while handling the turn.
        is_distractor: Whether the turn is deliberate noise carrying no real task.
        injected_failure: Tool fault to inject while handling this turn.
        injected_failure_tool: Which tool the fault applies to.
        contains_injection: Whether retrieved content for this turn carries an
            adversarial instruction the agent must refuse.
    """

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    role: TurnRole = TurnRole.USER
    content: str = Field(min_length=1)
    expected_behavior: str = ""
    introduced_facts: list[Fact] = Field(default_factory=list)
    updated_facts: list[Fact] = Field(default_factory=list)
    constraints_added: list[str] = Field(default_factory=list)
    constraints_removed: list[str] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    is_distractor: bool = False
    injected_failure: InjectedFailureType = InjectedFailureType.NONE
    injected_failure_tool: ToolName | None = None
    contains_injection: bool = False
    injection_payload: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_failure_target(self) -> ConversationTurn:
        if (
            self.injected_failure is not InjectedFailureType.NONE
            and self.injected_failure_tool is None
        ):
            raise ValueError(
                f"turn {self.turn_id}: injected_failure {self.injected_failure} "
                "requires injected_failure_tool"
            )
        return self


class SuccessCriterion(BaseModel):
    """One checkable condition for calling the session a success.

    Args:
        criterion_id: Stable identifier.
        description: Human-readable statement of the condition.
        evaluator: Name of the evaluator responsible for deciding it.
        weight: Relative importance within task-completion scoring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_id: str
    description: str
    evaluator: str
    weight: float = Field(default=1.0, gt=0)


class Scenario(BaseModel):
    """A complete adversarial multi-turn evaluation case.

    Args:
        scenario_id: Stable, content-derived identifier.
        name: Short human-readable title.
        description: What pressure this scenario applies and why it matters.
        category: Which failure mode it targets.
        difficulty: Derived pressure level.
        conversation_turns: The scripted turns, in order.
        initial_state: Workspace state the agent starts with.
        expected_facts: Canonical end-state facts the agent must hold.
        active_constraints: Every constraint referenced by the turns.
        required_tools: Tools that must be exercised at least once.
        expected_tool_sequence: Ordered tool names for sequence scoring.
        success_criteria: Conditions defining task completion.
        failure_conditions: Prose descriptions of what would count as failure.
        tags: Free-form labels for filtering.
        random_seed: Seed that produced this scenario; regeneration is reproducible.
        metadata: Generator provenance and derived statistics.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: ScenarioCategory
    difficulty: Difficulty
    conversation_turns: list[ConversationTurn] = Field(min_length=1)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    expected_facts: list[Fact] = Field(default_factory=list)
    active_constraints: list[Constraint] = Field(default_factory=list)
    required_tools: list[ToolName] = Field(default_factory=list)
    expected_tool_sequence: list[ToolName] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    random_seed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Number of scripted turns."""
        return len(self.conversation_turns)

    @property
    def user_turns(self) -> list[ConversationTurn]:
        """Only the turns spoken by the user."""
        return [t for t in self.conversation_turns if t.role is TurnRole.USER]

    def constraint_by_id(self, constraint_id: str) -> Constraint | None:
        """Look up a constraint by identifier."""
        for constraint in self.active_constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        return None

    def constraints_active_at(self, turn_index: int) -> list[Constraint]:
        """Every constraint in force at ``turn_index``."""
        return [c for c in self.active_constraints if c.is_active_at(turn_index)]

    def final_fact_values(self) -> dict[str, Fact]:
        """Resolve the end-state value of every fact after all updates.

        Later turns override earlier ones, which is precisely the behaviour a
        context-degradation scenario is probing: the agent must use the *latest*
        budget but the *original* launch date.
        """
        resolved: dict[str, Fact] = {}
        for turn in self.conversation_turns:
            for fact in turn.introduced_facts:
                resolved[fact.key] = fact
            for fact in turn.updated_facts:
                resolved[fact.key] = fact
        for fact in self.expected_facts:
            resolved.setdefault(fact.key, fact)
        return resolved

    @model_validator(mode="after")
    def _check_turn_indices(self) -> Scenario:
        indices = [t.turn_index for t in self.conversation_turns]
        if indices != sorted(indices):
            raise ValueError(f"scenario {self.scenario_id}: turn_index values must be ascending")
        if len(set(indices)) != len(indices):
            raise ValueError(f"scenario {self.scenario_id}: duplicate turn_index values")
        return self

    @model_validator(mode="after")
    def _check_constraint_references(self) -> Scenario:
        known = {c.constraint_id for c in self.active_constraints}
        for turn in self.conversation_turns:
            for cid in (*turn.constraints_added, *turn.constraints_removed):
                if cid not in known:
                    raise ValueError(
                        f"scenario {self.scenario_id}: turn {turn.turn_id} references "
                        f"unknown constraint {cid!r}"
                    )
        return self


class ScenarioSuite(BaseModel):
    """A named, versioned collection of scenarios.

    Args:
        suite_id: Stable identifier.
        name: Human-readable suite name.
        description: What the suite covers.
        generator_version: Version of the generator that produced it.
        seed: Master seed; regenerating with it reproduces the suite exactly.
        scenarios: The scenarios themselves.
    """

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    name: str
    description: str = ""
    generator_version: str = "1.0.0"
    seed: int = 42
    scenarios: list[Scenario] = Field(default_factory=list)

    def by_category(self) -> dict[ScenarioCategory, list[Scenario]]:
        """Group the suite's scenarios by category."""
        grouped: dict[ScenarioCategory, list[Scenario]] = {}
        for scenario in self.scenarios:
            grouped.setdefault(scenario.category, []).append(scenario)
        return grouped

    def by_difficulty(self) -> dict[Difficulty, list[Scenario]]:
        """Group the suite's scenarios by difficulty."""
        grouped: dict[Difficulty, list[Scenario]] = {}
        for scenario in self.scenarios:
            grouped.setdefault(scenario.difficulty, []).append(scenario)
        return grouped
