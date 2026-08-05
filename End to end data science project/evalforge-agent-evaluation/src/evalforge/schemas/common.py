"""Shared enumerations and primitive types used across every EvalForge schema.

These vocabularies are the backbone of the system: scenarios are tagged with them,
evaluators emit them, analytics group by them and the release report is organised
around them. Keeping them in one module means the failure taxonomy documented in
``docs/FAILURE_TAXONOMY.md`` has exactly one machine-readable definition.
"""

from __future__ import annotations

from enum import StrEnum


class ScenarioCategory(StrEnum):
    """Adversarial pressure a scenario is designed to apply.

    Each category isolates one way a multi-turn agent degrades. A suite is only
    meaningful when every category is represented, because an agent can be flawless
    at tool selection while still losing a deadline stated fifteen turns earlier.
    """

    CONTEXT_DEGRADATION = "context_degradation"
    INSTRUCTION_FORGETTING = "instruction_forgetting"
    GOAL_DRIFT = "goal_drift"
    CASCADING_ERRORS = "cascading_errors"
    TOOL_RELIABILITY = "tool_reliability"
    FAILURE_RECOVERY = "failure_recovery"
    PROMPT_INJECTION = "prompt_injection"
    LONG_SESSION_STRESS = "long_session_stress"


class Difficulty(StrEnum):
    """How much pressure a scenario applies, independent of category.

    Difficulty is a function of conversation length, fact count, constraint count,
    revision count and injected-failure count. It is assigned by the generator from
    those measurable properties rather than by hand, so it stays consistent.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class TurnRole(StrEnum):
    """Speaker of a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EventType(StrEnum):
    """Kinds of event recorded in a session trace.

    The trace is the single source of truth for evaluation; every evaluator reads
    events and nothing else. A dimension that cannot be derived from these event
    types is a dimension EvalForge cannot honestly score.
    """

    SESSION_START = "session_start"
    TURN_START = "turn_start"
    USER_MESSAGE = "user_message"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_RETRY = "tool_retry"
    VALIDATION_FAILURE = "validation_failure"
    STATE_TRANSITION = "state_transition"
    ASSISTANT_MESSAGE = "assistant_message"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    ARTIFACT_SAVED = "artifact_saved"
    TURN_END = "turn_end"
    SESSION_END = "session_end"


class EvaluationLevel(StrEnum):
    """Scope an evaluation result applies to.

    Turn-level results localise a failure; session-level results decide release
    readiness. Both are kept because "which turn broke" and "is the agent shippable"
    are different questions with different consumers.
    """

    TURN = "turn"
    SESSION = "session"


class Dimension(StrEnum):
    """Scored quality dimensions.

    These are the axes the weighted session score is built from. They map one-to-one
    onto the weights in ``configs/evaluation_rubrics.yaml``.
    """

    TASK_COMPLETION = "task_completion"
    CONTEXT_RETENTION = "context_retention"
    INSTRUCTION_ADHERENCE = "instruction_adherence"
    TOOL_RELIABILITY = "tool_reliability"
    RECOVERY_QUALITY = "recovery_quality"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    PLANNING_QUALITY = "planning_quality"
    USER_USEFULNESS = "user_usefulness"


class FailureCategory(StrEnum):
    """Machine-readable failure taxonomy.

    Every failing evaluation result carries one of these. Analytics groups by them,
    the dashboard charts them and the release report lists remediation against them.
    The prose definition of each lives in ``docs/FAILURE_TAXONOMY.md``.
    """

    # Context and memory
    FACT_LOST = "fact_lost"
    FACT_CORRUPTED = "fact_corrupted"
    STALE_FACT_USED = "stale_fact_used"
    DATE_LOST = "date_lost"

    # Instruction handling
    CONSTRAINT_VIOLATED = "constraint_violated"
    CONSTRAINT_FORGOTTEN = "constraint_forgotten"
    FORBIDDEN_CONTENT = "forbidden_content"
    REQUIRED_SECTION_MISSING = "required_section_missing"
    FORMAT_VIOLATION = "format_violation"

    # Goal management
    GOAL_DRIFT = "goal_drift"
    TASK_ABANDONED = "task_abandoned"
    OBJECTIVE_INCOMPLETE = "objective_incomplete"

    # Tool use
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    MISSING_TOOL_CALL = "missing_tool_call"
    WRONG_TOOL_ARGUMENT = "wrong_tool_argument"
    WRONG_TOOL_SEQUENCE = "wrong_tool_sequence"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    UNNECESSARY_TOOL_CALL = "unnecessary_tool_call"
    WRONG_ENTITY_SELECTED = "wrong_entity_selected"

    # Reliability and recovery
    RECOVERY_FAILED = "recovery_failed"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    CASCADING_ERROR = "cascading_error"

    # Truthfulness
    UNSUPPORTED_CLAIM = "unsupported_claim"
    FABRICATED_TOOL_RESULT = "fabricated_tool_result"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    INCORRECT_CALCULATION = "incorrect_calculation"

    # Safety
    PROMPT_INJECTION_FOLLOWED = "prompt_injection_followed"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    CONFIDENTIAL_DISCLOSURE = "confidential_disclosure"

    # Sentinel
    NONE = "none"


class Severity(StrEnum):
    """How badly a failure matters.

    ``CRITICAL`` is not merely "worse than major": a single critical failure blocks
    release regardless of the weighted score. See ADR-004.
    """

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ReleaseDecision(StrEnum):
    """Outcome of the release-readiness gate."""

    PASS = "pass"
    CONDITIONAL_PASS = "conditional_pass"
    FAIL = "fail"


class InjectedFailureType(StrEnum):
    """Tool-layer faults the harness can inject deterministically.

    Recovery behaviour is only measurable when failures are reproducible, so every
    injection is keyed off the run seed rather than wall-clock randomness.
    """

    NONE = "none"
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    PARTIAL_RESULT = "partial_result"
    INVALID_JSON = "invalid_json"
    MISSING_FIELD = "missing_field"
    CONFLICTING_DATA = "conflicting_data"
    STALE_DATA = "stale_data"
    TEMPORARY_ERROR = "temporary_error"
    INCORRECT_ENTITY = "incorrect_entity"
    INVALID_ARGUMENT_TYPE = "invalid_argument_type"
    UNAUTHORIZED_ACTION = "unauthorized_action"


class ToolName(StrEnum):
    """The productivity tool surface exposed to the agent under test."""

    SEARCH_DOCUMENTS = "search_documents"
    EXTRACT_REQUIREMENTS = "extract_requirements"
    CALCULATE_BUDGET = "calculate_budget"
    CREATE_PROJECT_PLAN = "create_project_plan"
    UPDATE_PROJECT_PLAN = "update_project_plan"
    DRAFT_EXECUTIVE_SUMMARY = "draft_executive_summary"
    DRAFT_STAKEHOLDER_EMAIL = "draft_stakeholder_email"
    SAVE_ARTIFACT = "save_artifact"


#: Failure categories that block release on a single occurrence (see ADR-004).
CRITICAL_FAILURE_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.PROMPT_INJECTION_FOLLOWED,
        FailureCategory.UNAUTHORIZED_ACTION,
        FailureCategory.FABRICATED_TOOL_RESULT,
        FailureCategory.DATE_LOST,
        FailureCategory.INCORRECT_CALCULATION,
        FailureCategory.CONFIDENTIAL_DISCLOSURE,
    }
)

#: Which dimension each failure category degrades, for score attribution.
FAILURE_DIMENSION_MAP: dict[FailureCategory, Dimension] = {
    FailureCategory.FACT_LOST: Dimension.CONTEXT_RETENTION,
    FailureCategory.FACT_CORRUPTED: Dimension.CONTEXT_RETENTION,
    FailureCategory.STALE_FACT_USED: Dimension.CONTEXT_RETENTION,
    FailureCategory.DATE_LOST: Dimension.CONTEXT_RETENTION,
    FailureCategory.CONSTRAINT_VIOLATED: Dimension.INSTRUCTION_ADHERENCE,
    FailureCategory.CONSTRAINT_FORGOTTEN: Dimension.INSTRUCTION_ADHERENCE,
    FailureCategory.FORBIDDEN_CONTENT: Dimension.INSTRUCTION_ADHERENCE,
    FailureCategory.REQUIRED_SECTION_MISSING: Dimension.INSTRUCTION_ADHERENCE,
    FailureCategory.FORMAT_VIOLATION: Dimension.INSTRUCTION_ADHERENCE,
    FailureCategory.GOAL_DRIFT: Dimension.TASK_COMPLETION,
    FailureCategory.TASK_ABANDONED: Dimension.TASK_COMPLETION,
    FailureCategory.OBJECTIVE_INCOMPLETE: Dimension.TASK_COMPLETION,
    FailureCategory.WRONG_TOOL_SELECTED: Dimension.TOOL_RELIABILITY,
    FailureCategory.MISSING_TOOL_CALL: Dimension.TOOL_RELIABILITY,
    FailureCategory.WRONG_TOOL_ARGUMENT: Dimension.TOOL_RELIABILITY,
    FailureCategory.WRONG_TOOL_SEQUENCE: Dimension.TOOL_RELIABILITY,
    FailureCategory.DUPLICATE_TOOL_CALL: Dimension.EFFICIENCY,
    FailureCategory.UNNECESSARY_TOOL_CALL: Dimension.EFFICIENCY,
    FailureCategory.WRONG_ENTITY_SELECTED: Dimension.TOOL_RELIABILITY,
    FailureCategory.RECOVERY_FAILED: Dimension.RECOVERY_QUALITY,
    FailureCategory.RETRY_LIMIT_EXCEEDED: Dimension.RECOVERY_QUALITY,
    FailureCategory.LATENCY_EXCEEDED: Dimension.EFFICIENCY,
    FailureCategory.CASCADING_ERROR: Dimension.RECOVERY_QUALITY,
    FailureCategory.UNSUPPORTED_CLAIM: Dimension.CONSISTENCY,
    FailureCategory.FABRICATED_TOOL_RESULT: Dimension.SAFETY,
    FailureCategory.INTERNAL_CONTRADICTION: Dimension.CONSISTENCY,
    FailureCategory.INCORRECT_CALCULATION: Dimension.TASK_COMPLETION,
    FailureCategory.PROMPT_INJECTION_FOLLOWED: Dimension.SAFETY,
    FailureCategory.UNAUTHORIZED_ACTION: Dimension.SAFETY,
    FailureCategory.CONFIDENTIAL_DISCLOSURE: Dimension.SAFETY,
}


def is_critical(category: FailureCategory) -> bool:
    """Return whether a single occurrence of ``category`` blocks release."""
    return category in CRITICAL_FAILURE_CATEGORIES


def dimension_for(category: FailureCategory) -> Dimension | None:
    """Return the scored dimension a failure category degrades, if any."""
    return FAILURE_DIMENSION_MAP.get(category)
