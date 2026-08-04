"""The evaluation engine.

Three evaluator families, kept structurally separate and never averaged into one another
(ADR-002): deterministic checks against the scenario contract, semantic similarity, and
an LLM judge. Deterministic results gate releases; the others are reported alongside.
"""

from __future__ import annotations

from evalforge.evaluators.aggregation import (
    SCORED_DIMENSIONS,
    aggregate_session,
    critical_failures,
    score_dimension,
)
from evalforge.evaluators.base import BaseEvaluator, EvaluationContext, Evaluator
from evalforge.evaluators.judge import (
    JUDGE_DIMENSIONS,
    Judge,
    JudgeEvaluator,
    JudgeResponse,
    LLMJudge,
    MockJudge,
    build_judge,
)
from evalforge.evaluators.registry import (
    DETERMINISTIC_EVALUATORS,
    SEMANTIC_EVALUATORS,
    build_evaluators,
    evaluate_session,
    evaluator_names,
)
from evalforge.evaluators.semantic import Embedder, LexicalEmbedder

__all__ = [
    "DETERMINISTIC_EVALUATORS",
    "JUDGE_DIMENSIONS",
    "SCORED_DIMENSIONS",
    "SEMANTIC_EVALUATORS",
    "BaseEvaluator",
    "Embedder",
    "EvaluationContext",
    "Evaluator",
    "Judge",
    "JudgeEvaluator",
    "JudgeResponse",
    "LLMJudge",
    "LexicalEmbedder",
    "MockJudge",
    "aggregate_session",
    "build_evaluators",
    "build_judge",
    "critical_failures",
    "evaluate_session",
    "evaluator_names",
    "score_dimension",
]
