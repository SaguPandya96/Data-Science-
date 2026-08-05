"""Context-retention evaluators.

These answer the question single-turn evaluation structurally cannot: does the agent
still know, at turn 20, what the user told it at turn 1 — and does it know the *current*
value rather than a superseded one?

Two independent signals are used. The workspace snapshot records what the agent believes;
the assistant text records what it said. Checking both matters because they fail
differently: an agent can hold a fact and misstate it, or drop it internally while a
stale phrase survives in prose.
"""

from __future__ import annotations

from evalforge.evaluators.base import (
    BaseEvaluator,
    dates_match,
    fact_matches,
    normalize_money,
    text_contains_fact,
)
from evalforge.evaluators.base import excerpt as _excerpt
from evalforge.schemas.common import Dimension, EvaluationLevel, FailureCategory, Severity
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Fact, Scenario
from evalforge.schemas.trace import SessionTrace


def _final_facts(scenario: Scenario) -> dict[str, Fact]:
    """Resolve every fact to its latest user-stated value."""
    return scenario.final_fact_values()


def _agent_value(trace: SessionTrace, key: str) -> str | None:
    """What the agent held for ``key`` at session end."""
    if trace.final_workspace is None:
        return None
    return trace.final_workspace.facts.get(key)


class FactRetentionEvaluator(BaseEvaluator):
    """Did the agent still hold every stated fact at the end of the session?"""

    name = "fact_retention"
    dimension = Dimension.CONTEXT_RETENTION

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score retention of each fact, and roll them up into a session score."""
        facts = _final_facts(scenario)
        if not facts:
            return []

        results: list[EvaluationResult] = []
        retained = 0
        lost: list[str] = []
        corrupted: list[str] = []

        # Corroborating text is the *final* turn's reply, never the whole session.
        # Searching the whole transcript would let a fact stated correctly on turn 0 and
        # lost by turn 20 still count as retained, which is precisely the failure this
        # evaluator exists to catch.
        final_text = trace.turns[-1].assistant_message if trace.turns else ""

        for key, fact in sorted(facts.items()):
            held = _agent_value(trace, key)
            in_workspace = held is not None and fact_matches(fact, held)
            # Text is a fallback for providers that do not report structured state; when
            # a workspace exists it is the authoritative record of what the agent held.
            ok = in_workspace or (
                trace.final_workspace is None and text_contains_fact(final_text, fact)
            )

            if ok:
                retained += 1
            elif held is None:
                lost.append(key)
            else:
                corrupted.append(key)

            if not ok and fact.critical:
                # Corruption is scored more harshly than loss: a confidently wrong value
                # reaches the user unchallenged, while a missing one usually surfaces.
                category = (
                    FailureCategory.FACT_LOST if held is None else FailureCategory.FACT_CORRUPTED
                )
                verb = "dropped" if held is None else "altered"
                results.append(
                    self.make_result(
                        scenario,
                        trace,
                        score=0.0,
                        passed=False,
                        suffix=f"fact::{key}",
                        evidence=[
                            Evidence(
                                kind="workspace_fact",
                                excerpt=f"{key}={held!r} (expected {fact.value!r})",
                                detail="Final workspace value at session end.",
                            )
                        ],
                        reasoning=(
                            f"Critical fact {key!r} was {verb}: expected "
                            f"{fact.value!r}, agent held {held!r}."
                        ),
                        failure_category=category,
                        severity=Severity.MAJOR,
                    )
                )

        score = retained / len(facts)
        summary = f"Retained {retained}/{len(facts)} facts."
        if lost:
            summary += f" Lost: {', '.join(lost)}."
        if corrupted:
            summary += f" Corrupted: {', '.join(corrupted)}."

        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                evidence=[
                    Evidence(
                        kind="workspace",
                        excerpt=_excerpt(str(dict(trace.final_workspace.facts)))
                        if trace.final_workspace
                        else "<no workspace>",
                        detail="Facts the agent held when the session ended.",
                    )
                ],
                reasoning=summary,
                failure_category=FailureCategory.FACT_LOST,
                severity=Severity.MAJOR,
                metadata={"lost": lost, "corrupted": corrupted, "total": len(facts)},
            )
        )
        return results


class UpdatedFactEvaluator(BaseEvaluator):
    """Did the agent use the *revised* value after the user changed it?

    Separated from plain retention because the remediation is different. Losing a fact is
    a memory problem; using the pre-revision value is an override-semantics problem, and
    it produces confident, plausible, wrong output.
    """

    name = "updated_fact"
    dimension = Dimension.CONTEXT_RETENTION

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score correct use of every revised fact."""
        revised: dict[str, tuple[Fact, list[str]]] = {}
        history: dict[str, list[str]] = {}

        for turn in scenario.conversation_turns:
            for fact in (*turn.introduced_facts, *turn.updated_facts):
                history.setdefault(fact.key, []).append(fact.value)
            for fact in turn.updated_facts:
                revised[fact.key] = (fact, history[fact.key][:-1])

        if not revised:
            return []

        correct = 0
        stale_keys: list[str] = []
        results: list[EvaluationResult] = []

        for key, (fact, superseded) in sorted(revised.items()):
            held = _agent_value(trace, key)
            uses_current = held is not None and fact_matches(fact, held)
            uses_superseded = held is not None and any(
                fact_matches(fact.model_copy(update={"value": old}), held) for old in superseded
            )

            if uses_current:
                correct += 1
            elif uses_superseded:
                stale_keys.append(key)
                results.append(
                    self.make_result(
                        scenario,
                        trace,
                        score=0.0,
                        passed=False,
                        suffix=f"stale::{key}",
                        evidence=[
                            Evidence(
                                kind="workspace_fact",
                                excerpt=f"{key}={held!r}; current value is {fact.value!r}",
                                detail=f"Superseded values seen: {superseded}",
                            )
                        ],
                        reasoning=(
                            f"Agent used the superseded value {held!r} for {key!r} after the "
                            f"user revised it to {fact.value!r}."
                        ),
                        failure_category=FailureCategory.STALE_FACT_USED,
                        severity=Severity.MAJOR,
                    )
                )

        score = correct / len(revised)
        results.append(
            self.make_result(
                scenario,
                trace,
                score=score,
                suffix="session",
                reasoning=(
                    f"Used the current value for {correct}/{len(revised)} revised facts."
                    + (f" Stale: {', '.join(stale_keys)}." if stale_keys else "")
                ),
                failure_category=FailureCategory.STALE_FACT_USED,
                severity=Severity.MAJOR,
                metadata={"revised_facts": sorted(revised), "stale": stale_keys},
            )
        )
        return results


class DateAccuracyEvaluator(BaseEvaluator):
    """Was every pinned deadline preserved exactly?

    Dates get their own evaluator, and a **critical** severity, because they propagate
    into commitments made to third parties. Scenarios pin them explicitly ("keep the
    original launch date"), so there is no ambiguity of intent to excuse a miss.
    """

    name = "date_accuracy"
    dimension = Dimension.CONTEXT_RETENTION

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Score preservation of each date fact across workspace, plan and artifacts."""
        date_facts = [fact for fact in _final_facts(scenario).values() if fact.kind.value == "date"]
        if not date_facts:
            return []

        results: list[EvaluationResult] = []
        correct = 0

        for fact in date_facts:
            held = _agent_value(trace, fact.key)
            plan_value = (trace.final_artifact or {}).get("launch_date")
            summary_value = None
            for turn in trace.turns:
                for call in turn.tool_calls:
                    if call.succeeded and call.result and "launch_date" in call.result:
                        summary_value = call.result["launch_date"]

            observed = [value for value in (held, plan_value, summary_value) if value]
            matches = [dates_match(fact.value, str(value)) for value in observed]
            ok = bool(matches) and all(matches)

            if ok:
                correct += 1
                continue

            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=0.0,
                    passed=False,
                    suffix=f"date::{fact.key}",
                    evidence=[
                        Evidence(
                            kind="date_comparison",
                            excerpt=f"expected {fact.value}; observed {observed or ['<absent>']}",
                            detail="Compared across workspace, plan and produced artifacts.",
                        )
                    ],
                    reasoning=(
                        f"Pinned date {fact.key!r} ({fact.value}) was not preserved. "
                        f"Observed: {observed or 'nothing'}."
                    ),
                    failure_category=FailureCategory.DATE_LOST,
                    severity=Severity.CRITICAL if fact.critical else Severity.MAJOR,
                )
            )

        results.append(
            self.make_result(
                scenario,
                trace,
                score=correct / len(date_facts),
                suffix="session",
                reasoning=f"Preserved {correct}/{len(date_facts)} pinned dates.",
                failure_category=FailureCategory.DATE_LOST,
                severity=Severity.MAJOR,
            )
        )
        return results


class BudgetCalculationEvaluator(BaseEvaluator):
    """Does every budget calculation follow from its own inputs?

    Exact arithmetic, which is why it can carry **critical** severity with no
    false-positive risk: there is no paraphrase or judgement involved.
    """

    name = "budget_calculation"
    dimension = Dimension.TASK_COMPLETION
    level = EvaluationLevel.TURN

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Recompute each budget call and compare against what the tool returned."""
        results: list[EvaluationResult] = []
        checked = 0
        wrong = 0

        for call in trace.all_tool_calls:
            if call.tool_name.value != "calculate_budget" or not call.succeeded:
                continue
            result = call.result or {}
            if result.get("_injected"):
                # The harness deliberately corrupted this payload; scoring the agent for
                # the harness's own fault would be measuring the wrong thing.
                continue

            line_items = call.arguments.get("line_items", [])
            expected_allocated = sum(
                normalize_money(item.get("amount")) or 0.0
                for item in line_items
                if isinstance(item, dict)
            )
            reported = normalize_money(result.get("allocated"))
            checked += 1

            if reported is None or abs(reported - expected_allocated) > 0.01:
                wrong += 1
                results.append(
                    self.make_result(
                        scenario,
                        trace,
                        score=0.0,
                        passed=False,
                        turn_index=call.turn_index,
                        suffix=f"budget::{call.tool_call_id}",
                        evidence=[
                            Evidence(
                                kind="tool_call",
                                turn_index=call.turn_index,
                                excerpt=(
                                    f"allocated={reported}; line items sum to {expected_allocated}"
                                ),
                                detail="Recomputed from the arguments the agent supplied.",
                            )
                        ],
                        reasoning=(
                            f"Budget total does not follow from its inputs: reported {reported}, "
                            f"inputs sum to {expected_allocated:.2f}."
                        ),
                        failure_category=FailureCategory.INCORRECT_CALCULATION,
                        severity=Severity.CRITICAL,
                    )
                )

        if not checked:
            return results

        results.append(
            self.make_result(
                scenario,
                trace,
                score=(checked - wrong) / checked,
                suffix="session",
                level=EvaluationLevel.SESSION,
                reasoning=(
                    f"{checked - wrong}/{checked} budget calculations were arithmetically sound."
                ),
                failure_category=FailureCategory.INCORRECT_CALCULATION,
                severity=Severity.MAJOR,
            )
        )
        return results
