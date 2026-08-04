"""Evaluator construction and the scoring entry point.

One function builds the evaluator set for a run, and one function applies it. Keeping
both here means the CLI, the orchestrator, the dashboard and the tests all score a
session identically — a property the alignment analysis depends on, since it compares
human labels against *the* automated verdict, not one of several.
"""

from __future__ import annotations

from typing import Any

from evalforge.config import EvalForgeConfig
from evalforge.evaluators.base import Evaluator
from evalforge.evaluators.context import (
    BudgetCalculationEvaluator,
    DateAccuracyEvaluator,
    FactRetentionEvaluator,
    UpdatedFactEvaluator,
)
from evalforge.evaluators.instructions import (
    ForbiddenContentEvaluator,
    PersistentConstraintEvaluator,
    RequiredSectionEvaluator,
)
from evalforge.evaluators.integrity import (
    CascadingErrorEvaluator,
    ContradictionEvaluator,
    GoalDriftEvaluator,
    UnsupportedClaimEvaluator,
)
from evalforge.evaluators.judge import Judge, JudgeEvaluator, build_judge
from evalforge.evaluators.reliability import (
    LatencyThresholdEvaluator,
    RecoveryEvaluator,
    RetryLimitEvaluator,
)
from evalforge.evaluators.safety import PromptInjectionEvaluator, UnauthorizedActionEvaluator
from evalforge.evaluators.semantic import (
    Embedder,
    GoalAlignmentEvaluator,
    MeaningPreservationEvaluator,
)
from evalforge.evaluators.tool_use import (
    DuplicateToolCallEvaluator,
    RequiredStepEvaluator,
    ToolArgumentEvaluator,
    ToolSelectionEvaluator,
    ToolSequenceEvaluator,
)
from evalforge.logging_config import get_logger
from evalforge.schemas.evaluation import EvaluationResult
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace

logger = get_logger(__name__)

#: The deterministic evaluators, in the order their results appear in a report.
DETERMINISTIC_EVALUATORS: tuple[type[Any], ...] = (
    FactRetentionEvaluator,
    UpdatedFactEvaluator,
    DateAccuracyEvaluator,
    BudgetCalculationEvaluator,
    ForbiddenContentEvaluator,
    RequiredSectionEvaluator,
    PersistentConstraintEvaluator,
    ToolSelectionEvaluator,
    ToolArgumentEvaluator,
    ToolSequenceEvaluator,
    DuplicateToolCallEvaluator,
    RequiredStepEvaluator,
    RecoveryEvaluator,
    RetryLimitEvaluator,
    LatencyThresholdEvaluator,
    UnsupportedClaimEvaluator,
    ContradictionEvaluator,
    GoalDriftEvaluator,
    CascadingErrorEvaluator,
    PromptInjectionEvaluator,
    UnauthorizedActionEvaluator,
)

SEMANTIC_EVALUATORS: tuple[type[Any], ...] = (
    GoalAlignmentEvaluator,
    MeaningPreservationEvaluator,
)


def build_evaluators(
    config: EvalForgeConfig,
    embedder: Embedder | None = None,
    judge: Judge | None = None,
) -> list[Evaluator]:
    """Construct the evaluator set the configuration asks for.

    Args:
        config: Effective configuration.
        embedder: Optional embedding backend for semantic evaluators.
        judge: Optional judge implementation, defaulting to the configured one.
    """
    evaluators: list[Evaluator] = []

    if config.evaluation.run_deterministic:
        evaluators.extend(cls(config) for cls in DETERMINISTIC_EVALUATORS)

    if config.evaluation.run_semantic:
        evaluators.extend(cls(config, embedder) for cls in SEMANTIC_EVALUATORS)

    if config.evaluation.run_judge:
        evaluators.append(JudgeEvaluator(config, judge or build_judge(config)))

    return evaluators


def evaluate_session(
    scenario: Scenario,
    trace: SessionTrace,
    evaluators: list[Evaluator],
) -> list[EvaluationResult]:
    """Run every evaluator over one session.

    An evaluator that raises is logged and skipped rather than aborting the run. A
    single buggy check should cost one dimension's coverage, not a 150-session
    evaluation — and the skipped evaluator is visible in the report because its results
    are simply absent.
    """
    results: list[EvaluationResult] = []
    for evaluator in evaluators:
        try:
            results.extend(evaluator.evaluate(scenario, trace))
        except Exception as exc:
            logger.error(
                "evaluator_failed",
                evaluator=evaluator.name,
                scenario_id=scenario.scenario_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
    return results


def evaluator_names(evaluators: list[Evaluator]) -> list[str]:
    """Names of the evaluators in a set, for reporting."""
    return [evaluator.name for evaluator in evaluators]
