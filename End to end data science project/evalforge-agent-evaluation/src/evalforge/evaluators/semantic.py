"""Semantic evaluators.

Some questions need meaning rather than string equality: did the reply preserve what the
user asked for, is it relevant, does it align with the stated goal. These are scored
here — but never gated on (ADR-002), because the default backend is lexical and lexical
similarity is a weak proxy for meaning.

The design point is the fallback. An :class:`Embedder` protocol lets a real embedding
model be injected, and a token-overlap implementation runs when none is available. The
core system therefore works with no model, no network and no extra dependency, and the
report labels which backend produced the numbers rather than quietly implying more
precision than exists.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from evalforge.evaluators.base import BaseEvaluator, excerpt, similarity
from evalforge.schemas.common import Dimension, FailureCategory, Severity
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words too common to carry meaning in overlap scoring.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "as",
        "it",
        "its",
        "it's",
        "we",
        "you",
        "i",
        "he",
        "she",
        "they",
        "them",
        "us",
        "our",
        "your",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "not",
        "no",
        "yes",
        "so",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "now",
    ]
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a vector."""

    name: str

    def embed(self, text: str) -> list[float]:
        """Return an embedding for ``text``."""
        ...


class LexicalEmbedder:
    """Bag-of-words fallback embedder.

    Deliberately simple and deterministic. It is not a good semantic model and is not
    presented as one; it exists so the semantic code path runs offline and so the
    interface is exercised in CI.
    """

    name = "lexical-overlap"

    def embed(self, text: str) -> list[float]:
        """Not meaningful in isolation; :meth:`similarity` is the real interface."""
        counts = Counter(tokenize(text))
        return [float(counts[token]) for token in sorted(counts)]

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity over token counts, blended with sequence similarity.

        The blend matters: pure token overlap ignores word order entirely, and pure
        sequence similarity over-weights shared boilerplate. Neither alone behaves
        sensibly on short project prose.
        """
        left_tokens = Counter(tokenize(left))
        right_tokens = Counter(tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0

        shared = set(left_tokens) & set(right_tokens)
        dot = sum(left_tokens[token] * right_tokens[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
        right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
        cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
        return 0.7 * cosine + 0.3 * similarity(left, right)


def tokenize(text: str) -> list[str]:
    """Lower-case content tokens, stopwords removed."""
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


class SemanticEvaluator(BaseEvaluator):
    """Base for evaluators that compare meaning rather than exact values."""

    kind = "semantic"

    #: Semantic results report a score but never assert a failure.
    #:
    #: The default backend is lexical, and lexical overlap between a correct reply and
    #: the prose describing what was expected routinely sits below 0.2 — vocabulary
    #: difference, not error. Any threshold above zero would therefore label correct
    #: sessions as failures and put a proxy measure's noise into the failure taxonomy
    #: the release report groups by. The score stays visible as a diagnostic; the
    #: verdict is left to the deterministic checks (ADR-002).
    SEMANTIC_PASS_THRESHOLD = 0.0

    def threshold(self) -> float:
        """Semantic evaluators never fail a session; they only report a score."""
        return self.SEMANTIC_PASS_THRESHOLD

    def __init__(self, config, embedder: Embedder | None = None) -> None:  # type: ignore[no-untyped-def]
        """Construct with an optional embedder, defaulting to the lexical fallback."""
        super().__init__(config)
        self.embedder = embedder or LexicalEmbedder()

    def compare(self, left: str, right: str) -> float:
        """Similarity in ``[0, 1]`` using whichever backend is configured."""
        backend = self.embedder
        if isinstance(backend, LexicalEmbedder):
            return backend.similarity(left, right)
        left_vector = backend.embed(left)
        right_vector = backend.embed(right)
        return _cosine(left_vector, right_vector)

    @property
    def backend_name(self) -> str:
        """Which similarity backend produced these scores."""
        return self.embedder.name


def _cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class GoalAlignmentEvaluator(SemanticEvaluator):
    """Does each reply address what the turn actually asked for?"""

    name = "goal_alignment"
    dimension = Dimension.TASK_COMPLETION

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Compare each reply against the turn's expected behaviour."""
        scored: list[float] = []
        worst: tuple[float, int, str] | None = None

        for turn in scenario.conversation_turns:
            if turn.is_distractor or not turn.expected_behavior:
                continue
            record = trace.turn_at(turn.turn_index)
            if record is None or not record.assistant_message:
                continue
            score = self.compare(turn.expected_behavior, record.assistant_message)
            scored.append(score)
            if worst is None or score < worst[0]:
                worst = (score, turn.turn_index, record.assistant_message)

        if not scored:
            return []

        mean = sum(scored) / len(scored)
        evidence = []
        if worst:
            evidence.append(
                Evidence(
                    kind="assistant_message",
                    turn_index=worst[1],
                    excerpt=excerpt(worst[2]),
                    detail=f"Lowest alignment in the session: {worst[0]:.2f}",
                )
            )

        return [
            self.make_result(
                scenario,
                trace,
                score=mean,
                suffix="session",
                evidence=evidence,
                reasoning=(
                    f"Mean goal alignment {mean:.2f} across {len(scored)} task turns "
                    f"(backend: {self.backend_name})."
                ),
                failure_category=FailureCategory.GOAL_DRIFT,
                severity=Severity.MINOR,
                # Confidence is capped well below 1.0: the default backend is lexical,
                # and reporting full confidence in a proxy measure would be dishonest.
                confidence=0.55,
                metadata={"backend": self.backend_name, "turns_scored": len(scored)},
            )
        ]


class MeaningPreservationEvaluator(SemanticEvaluator):
    """Does the final summary still say what the plan says?"""

    name = "meaning_preservation"
    dimension = Dimension.CONSISTENCY

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Compare the executive summary against the plan it was derived from."""
        summaries = [
            call.result
            for call in trace.all_tool_calls
            if call.tool_name.value == "draft_executive_summary" and call.succeeded and call.result
        ]
        plan = trace.final_artifact
        if not summaries or not plan:
            return []

        summary_text = str(summaries[-1].get("text", ""))
        plan_text = " ".join(
            [
                str(plan.get("project_name", "")),
                str(plan.get("launch_date", "")),
                str(plan.get("budget", "")),
                *[str(phase.get("name", "")) for phase in plan.get("phases", []) or []],
            ]
        )
        score = self.compare(plan_text, summary_text)

        return [
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                evidence=[
                    Evidence(
                        kind="executive_summary",
                        excerpt=excerpt(summary_text),
                        detail=f"Compared against plan content (backend: {self.backend_name}).",
                    )
                ],
                reasoning=f"Summary/plan meaning overlap {score:.2f}.",
                failure_category=FailureCategory.INTERNAL_CONTRADICTION,
                severity=Severity.MINOR,
                confidence=0.55,
                metadata={"backend": self.backend_name},
            )
        ]
