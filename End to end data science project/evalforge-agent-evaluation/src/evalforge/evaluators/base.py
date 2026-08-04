"""Evaluator protocol and the comparison primitives every evaluator shares.

An evaluator reads a ``(Scenario, SessionTrace)`` pair and returns
:class:`EvaluationResult` objects. It may not mutate either input, may not perform I/O,
and must be deterministic — the same pair must always produce the same verdict, or the
regression gate becomes meaningless.

The normalisation helpers live here rather than in each evaluator because comparison
semantics are a *policy*, not an implementation detail. "Is ``$15,000`` the same as
``15000.00``?" must have one answer across the whole system, and if it changes, it
changes in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol, runtime_checkable

from evalforge.config import EvalForgeConfig
from evalforge.ids import stable_id
from evalforge.schemas.common import (
    Dimension,
    EvaluationLevel,
    FailureCategory,
    Severity,
)
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Fact, FactKind, Scenario
from evalforge.schemas.trace import SessionTrace

#: Month names accepted when normalising a spoken date such as "September 15".
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}  # fmt: skip

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SPOKEN_DATE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"\$?\s*(\d[\d,]*(?:\.\d+)?)")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Case-fold and collapse whitespace for tolerant string comparison."""
    return _WHITESPACE.sub(" ", value.strip().lower())


def normalize_money(value: str | float | int | None) -> float | None:
    """Parse a monetary value, tolerating symbols, separators and currency codes.

    Returns ``None`` when the input contains no number, so callers can distinguish
    "absent" from "zero" — a distinction that matters when a missing budget and a
    zero budget are different failures.
    """
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    match = _MONEY.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def normalize_date(value: str | None) -> str | None:
    """Normalise a date to ISO ``YYYY-MM-DD``, or ``None`` if none is present.

    Both ISO and spoken forms are accepted because the user types "September 15" and
    the tools store ``2026-09-15``; treating those as different would produce a
    date-loss failure every time an agent behaved correctly.
    """
    if not value:
        return None
    text = str(value)
    if iso := _ISO_DATE.search(text):
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    if spoken := _SPOKEN_DATE.search(text):
        month = _MONTHS[spoken.group(1).lower()]
        day = int(spoken.group(2))
        # Year is unknown from a spoken form; a month-day key is still enough to detect
        # a *changed* date, which is what the evaluator is actually asking.
        return f"__-{month:02d}-{day:02d}"
    return None


def dates_match(expected: str, observed: str) -> bool:
    """Whether two date strings refer to the same day, across ISO and spoken forms."""
    left = normalize_date(expected)
    right = normalize_date(observed)
    if left is None or right is None:
        return False
    if left.startswith("__") or right.startswith("__"):
        return left[-6:] == right[-6:]
    return left == right


def fact_matches(fact: Fact, observed: str | None) -> bool:
    """Whether ``observed`` is a correct statement of ``fact``.

    Comparison is chosen by fact kind: dates and money are compared by value, text and
    entities by normalised containment. Using containment for text avoids penalising an
    agent for saying "the Analytics Dashboard" instead of "Analytics Dashboard".
    """
    if observed is None:
        return False
    if fact.kind is FactKind.DATE:
        return dates_match(fact.value, observed) or any(
            dates_match(alias, observed) for alias in fact.aliases
        )
    if fact.kind in {FactKind.MONEY, FactKind.NUMBER}:
        expected = normalize_money(fact.value)
        actual = normalize_money(observed)
        if expected is None or actual is None:
            return False
        return abs(expected - actual) < 0.01
    candidates = [fact.value, *fact.aliases]
    haystack = normalize_text(observed)
    return any(normalize_text(candidate) in haystack for candidate in candidates)


def text_contains_fact(text: str, fact: Fact) -> bool:
    """Whether assistant prose states ``fact`` correctly."""
    if not text:
        return False
    if fact.kind is FactKind.DATE:
        candidates = [*_iso_strings(text), *_spoken_strings(text)]
        return any(dates_match(fact.value, candidate) for candidate in candidates)
    if fact.kind in {FactKind.MONEY, FactKind.NUMBER}:
        target = normalize_money(fact.value)
        if target is None:
            return False
        return any(
            abs(value - target) < 0.01
            for value in (normalize_money(m) or 0.0 for m in _MONEY.findall(text.replace(",", "")))
        )
    return fact_matches(fact, text)


def _iso_strings(text: str) -> list[str]:
    """Every ISO date appearing in ``text``."""
    return ["-".join(match) for match in _ISO_DATE.findall(text)]


def _spoken_strings(text: str) -> list[str]:
    """Every spoken date appearing in ``text``."""
    return [f"{month} {day}" for month, day in _SPOKEN_DATE.findall(text)]


def lcs_ratio(expected: list[str], actual: list[str]) -> float:
    """Longest-common-subsequence overlap of two sequences, in ``[0, 1]``.

    Used for tool-order scoring. Exact equality would punish a
    differently-interleaved but dependency-respecting order, which is not a real defect.
    """
    if not expected:
        return 1.0
    matcher = SequenceMatcher(None, expected, actual, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(1.0, matched / len(expected))


def similarity(left: str, right: str) -> float:
    """Normalised lexical similarity of two strings, in ``[0, 1]``."""
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def contains_phrase(text: str, phrase: str) -> bool:
    """Whether ``phrase`` appears in ``text``, ignoring case and whitespace shape."""
    if not phrase:
        return False
    return normalize_text(phrase) in normalize_text(text)


@dataclass
class EvaluationContext:
    """Shared state handed to every evaluator invocation."""

    scenario: Scenario
    trace: SessionTrace
    config: EvalForgeConfig

    @property
    def run_id(self) -> str:
        """Run this evaluation belongs to."""
        return self.trace.run_id


class BaseEvaluator:
    """Convenience base providing result construction and threshold lookup.

    Implementing :class:`Evaluator` directly is fine; this class exists so the twenty
    deterministic evaluators do not each repeat identifier minting and threshold logic.
    """

    name: str = "base"
    kind: str = "deterministic"
    dimension: Dimension = Dimension.TASK_COMPLETION
    level: EvaluationLevel = EvaluationLevel.SESSION

    def __init__(self, config: EvalForgeConfig) -> None:
        self.config = config

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score one session. Overridden by every concrete evaluator."""
        raise NotImplementedError

    def threshold(self) -> float:
        """Pass threshold for this evaluator's dimension."""
        return self.config.rubric.pass_thresholds.get(self.dimension.value, 0.8)

    def make_result(
        self,
        scenario: Scenario,
        trace: SessionTrace,
        score: float,
        *,
        passed: bool | None = None,
        dimension: Dimension | None = None,
        turn_index: int | None = None,
        turn_id: str | None = None,
        evidence: list[Evidence] | None = None,
        reasoning: str = "",
        failure_category: FailureCategory = FailureCategory.NONE,
        severity: Severity = Severity.INFO,
        confidence: float = 1.0,
        level: EvaluationLevel | None = None,
        suffix: str = "",
        metadata: dict[str, object] | None = None,
    ) -> EvaluationResult:
        """Build a result with a stable identifier and consistent defaults."""
        resolved_score = max(0.0, min(1.0, score))
        resolved_pass = passed if passed is not None else resolved_score >= self.threshold()
        return EvaluationResult(
            evaluation_id=stable_id(
                "evaluation", trace.run_id, trace.session_id, self.name, suffix or str(turn_index)
            ),
            run_id=trace.run_id,
            scenario_id=scenario.scenario_id,
            session_id=trace.session_id,
            turn_id=turn_id,
            turn_index=turn_index,
            evaluator_name=self.name,
            evaluator_kind=self.kind,
            evaluation_level=level or self.level,
            dimension=dimension or self.dimension,
            score=resolved_score,
            passed=resolved_pass,
            confidence=confidence,
            evidence=evidence or [],
            reasoning_summary=reasoning,
            failure_category=failure_category if not resolved_pass else FailureCategory.NONE,
            severity=severity if not resolved_pass else Severity.INFO,
            metadata=dict(metadata or {}),
        )


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can score a session."""

    name: str
    kind: str

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Return zero or more results for this session."""
        ...


def excerpt(text: str, limit: int = 220) -> str:
    """Trim text for use as evidence, keeping results readable in the dashboard."""
    collapsed = _WHITESPACE.sub(" ", text.strip())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"
