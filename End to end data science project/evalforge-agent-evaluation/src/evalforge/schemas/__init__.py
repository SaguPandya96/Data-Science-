"""Typed domain models validated at every system boundary.

Nothing is persisted, loaded or passed between layers as a bare dict. Boundary
validation is what allows a trace written by one version of the agent to be scored by a
later version of the evaluators without silent field drift.
"""

from __future__ import annotations

from evalforge.schemas.annotation import (
    AgreementStatistic,
    AlignmentReport,
    DimensionRating,
    HumanAnnotation,
)
from evalforge.schemas.common import (
    CRITICAL_FAILURE_CATEGORIES,
    Difficulty,
    Dimension,
    EvaluationLevel,
    EventType,
    FailureCategory,
    InjectedFailureType,
    ReleaseDecision,
    ScenarioCategory,
    Severity,
    ToolName,
    TurnRole,
    dimension_for,
    is_critical,
)
from evalforge.schemas.evaluation import (
    ComparisonReport,
    DimensionScore,
    EvaluationResult,
    Evidence,
    MetricDelta,
    ReleaseReadinessReport,
    RunSummary,
    SessionSummary,
    ThresholdCheck,
)
from evalforge.schemas.scenario import (
    Constraint,
    ConstraintKind,
    ConversationTurn,
    ExpectedToolCall,
    Fact,
    FactKind,
    Scenario,
    ScenarioSuite,
    SuccessCriterion,
)
from evalforge.schemas.trace import (
    SessionTrace,
    TokenUsage,
    ToolCall,
    TraceEvent,
    TurnRecord,
    WorkspaceSnapshot,
)

__all__ = [
    "CRITICAL_FAILURE_CATEGORIES",
    "AgreementStatistic",
    "AlignmentReport",
    "ComparisonReport",
    "Constraint",
    "ConstraintKind",
    "ConversationTurn",
    "Difficulty",
    "Dimension",
    "DimensionRating",
    "DimensionScore",
    "EvaluationLevel",
    "EvaluationResult",
    "EventType",
    "Evidence",
    "ExpectedToolCall",
    "Fact",
    "FactKind",
    "FailureCategory",
    "HumanAnnotation",
    "InjectedFailureType",
    "MetricDelta",
    "ReleaseDecision",
    "ReleaseReadinessReport",
    "RunSummary",
    "Scenario",
    "ScenarioCategory",
    "ScenarioSuite",
    "SessionSummary",
    "SessionTrace",
    "Severity",
    "SuccessCriterion",
    "ThresholdCheck",
    "TokenUsage",
    "ToolCall",
    "ToolName",
    "TraceEvent",
    "TurnRecord",
    "TurnRole",
    "WorkspaceSnapshot",
    "dimension_for",
    "is_critical",
]
