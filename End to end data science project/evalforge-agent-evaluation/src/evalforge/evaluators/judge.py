"""LLM-as-a-judge with reliability controls.

The judge is treated as an unreliable instrument, and the controls follow from that:

- **Structured output only.** Responses parse into a Pydantic model; unparseable output
  is retried and then raises rather than being coerced into a number.
- **Evidence required.** A verdict with no trace excerpt is rejected. This is the single
  most effective control — it forces the judge to point at something real instead of
  producing a plausible-sounding score.
- **Multi-sample, median-aggregated.** Median rather than mean because judge outputs
  have outliers and the mean chases them.
- **Provenance recorded.** Judge model and prompt version land on every result; a judge
  score is only interpretable relative to the judge that produced it.
- **Never gates a release.** No judge result may carry critical severity (ADR-002/004).

``MockJudge`` implements the same interface deterministically so the whole judging code
path — sampling, aggregation, evidence validation, low-confidence flagging — is
exercised in CI with no credentials.
"""

from __future__ import annotations

import json
import statistics
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalforge.config import EvalForgeConfig
from evalforge.evaluators.base import BaseEvaluator, excerpt
from evalforge.exceptions import JudgeError
from evalforge.logging_config import get_logger
from evalforge.providers.base import Message, ModelProvider, ModelRequest
from evalforge.rng import seeded_random
from evalforge.schemas.common import Dimension, FailureCategory, Severity
from evalforge.schemas.evaluation import EvaluationResult, Evidence
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace

logger = get_logger(__name__)

JUDGE_PROMPT_VERSION = "v1"

#: Dimensions the judge scores. Each is one a deterministic rule cannot reach.
JUDGE_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension.TASK_COMPLETION,
    Dimension.CONTEXT_RETENTION,
    Dimension.INSTRUCTION_ADHERENCE,
    Dimension.PLANNING_QUALITY,
    Dimension.RECOVERY_QUALITY,
    Dimension.USER_USEFULNESS,
    Dimension.CONSISTENCY,
)

JUDGE_SYSTEM_PROMPT = """You are evaluating a complete multi-turn agent session.

You will receive the scenario contract and the full session transcript including tool
calls. Score each requested dimension from 0.0 to 1.0.

Rules:
- Every score must be justified by a quoted excerpt from the transcript. A verdict with
  no evidence will be rejected.
- Judge the session as a whole, not the last message.
- Do not reward length. A short, correct answer outranks a long, hedged one.
- Report low confidence when the transcript does not settle the question.

Respond with a single JSON object and nothing else:
{"scores": [{"dimension": str, "score": float, "passed": bool, "confidence": float,
             "evidence": [str], "reasoning_summary": str,
             "failure_category": str, "severity": str}]}
"""


class JudgeVerdict(BaseModel):
    """One judged dimension, as returned by the model."""

    model_config = ConfigDict(extra="ignore")

    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool = True
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    failure_category: str = "none"
    severity: str = "info"


class JudgeResponse(BaseModel):
    """A full judge reply."""

    model_config = ConfigDict(extra="ignore")

    scores: list[JudgeVerdict] = Field(default_factory=list)


@runtime_checkable
class Judge(Protocol):
    """Anything that can judge a session."""

    name: str
    model: str
    prompt_version: str

    def judge(self, scenario: Scenario, trace: SessionTrace) -> JudgeResponse:
        """Return one sampled judgement of the session."""
        ...


def render_transcript(scenario: Scenario, trace: SessionTrace, max_turns: int = 40) -> str:
    """Render a session for judging.

    Tool calls are included with their arguments and outcomes because a judge scoring
    "planning quality" from assistant prose alone is scoring writing, not planning.
    """
    lines = [
        f"SCENARIO: {scenario.name}",
        f"CATEGORY: {scenario.category.value}  DIFFICULTY: {scenario.difficulty.value}",
        f"DESCRIPTION: {scenario.description}",
        "",
        "ACTIVE CONSTRAINTS:",
    ]
    lines.extend(
        f"  - [{c.constraint_id}] {c.description} (from turn {c.turn_added})"
        for c in scenario.active_constraints
    )
    lines.append("")
    lines.append("EXPECTED END-STATE FACTS:")
    lines.extend(
        f"  - {key} = {fact.value}" for key, fact in sorted(scenario.final_fact_values().items())
    )
    lines.append("")
    lines.append("TRANSCRIPT:")

    for turn in trace.turns[:max_turns]:
        lines.append(f"[turn {turn.turn_index}] USER: {turn.user_message}")
        for call in turn.tool_calls:
            outcome = "ok" if call.succeeded else f"ERROR {call.error_type}"
            lines.append(
                f"    TOOL {call.tool_name.value}({_compact(call.arguments)}) -> {outcome}"
            )
        lines.append(f"[turn {turn.turn_index}] ASSISTANT: {turn.assistant_message}")
    return "\n".join(lines)


def _compact(arguments: dict[str, Any], limit: int = 160) -> str:
    """Render tool arguments compactly enough to fit in a prompt."""
    rendered = json.dumps(arguments, default=str, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


class MockJudge:
    """Deterministic offline judge.

    It derives verdicts from observable trace properties — retention, constraint
    handling, tool outcomes, recovery behaviour — plus small seeded noise, so that
    aggregation and disagreement analysis have realistic variance to work with. It is a
    stand-in for a model, not a model, and every report says so.
    """

    name = "mock-judge"

    def __init__(self, model: str = "mock-judge-v1", seed: int = 42) -> None:
        self.model = model
        self.prompt_version = JUDGE_PROMPT_VERSION
        self.seed = seed
        self._sample = 0

    def judge(self, scenario: Scenario, trace: SessionTrace) -> JudgeResponse:
        """Produce one sampled judgement."""
        self._sample += 1
        rng = seeded_random(self.seed, trace.session_id, "judge", self._sample)

        facts = scenario.final_fact_values()
        held = trace.final_workspace.facts if trace.final_workspace else {}
        retention = sum(1 for key in facts if key in held) / len(facts) if facts else 1.0
        calls = trace.all_tool_calls
        tool_health = sum(1 for call in calls if call.succeeded) / len(calls) if calls else 1.0
        failures = [call for call in calls if not call.succeeded]
        recovered = sum(
            1
            for call in failures
            if any(other.tool_name is call.tool_name and other.succeeded for other in calls)
        )
        recovery = recovered / len(failures) if failures else 1.0
        completion = 1.0 if trace.final_artifact else 0.4
        drift_turns = sum(
            1
            for turn in scenario.conversation_turns
            if turn.expected_tool_calls
            and not (trace.turn_at(turn.turn_index) or _EmptyTurn()).tool_calls
        )
        drift_penalty = min(0.5, 0.12 * drift_turns)

        base = {
            Dimension.TASK_COMPLETION: completion - drift_penalty,
            Dimension.CONTEXT_RETENTION: retention,
            Dimension.INSTRUCTION_ADHERENCE: 0.5 + 0.5 * retention,
            Dimension.PLANNING_QUALITY: 0.45 + 0.55 * tool_health,
            Dimension.RECOVERY_QUALITY: recovery,
            Dimension.USER_USEFULNESS: 0.35 + 0.4 * completion + 0.25 * retention,
            Dimension.CONSISTENCY: 0.5 + 0.5 * tool_health,
        }

        verdicts: list[JudgeVerdict] = []
        for dimension, raw in base.items():
            # Seeded jitter models real judge variance, which is what multi-sample
            # aggregation exists to absorb. Without it, median-of-three is a no-op and
            # the aggregation code would never be genuinely exercised.
            score = max(0.0, min(1.0, raw + rng.uniform(-0.08, 0.08)))
            passed = score >= 0.7
            verdicts.append(
                JudgeVerdict(
                    dimension=dimension.value,
                    score=round(score, 4),
                    passed=passed,
                    confidence=round(0.55 + 0.35 * tool_health, 3),
                    evidence=[excerpt(trace.turns[-1].assistant_message)] if trace.turns else [],
                    reasoning_summary=(
                        f"{dimension.value.replace('_', ' ')} judged at {score:.2f} from "
                        f"retention {retention:.2f}, tool health {tool_health:.2f}, "
                        f"recovery {recovery:.2f}."
                    ),
                    failure_category="none" if passed else _category_for(dimension),
                    # Judges never emit critical severity: a categorical release blocker
                    # must rest on an exact check, not a model's opinion (ADR-004).
                    severity="info" if passed else "major",
                )
            )
        return JudgeResponse(scores=verdicts)


class _EmptyTurn:
    """Null object so drift counting does not branch on ``None``."""

    tool_calls: list[Any] = []


def _category_for(dimension: Dimension) -> str:
    """Map a judged dimension onto a taxonomy label."""
    return {
        Dimension.TASK_COMPLETION: FailureCategory.OBJECTIVE_INCOMPLETE.value,
        Dimension.CONTEXT_RETENTION: FailureCategory.FACT_LOST.value,
        Dimension.INSTRUCTION_ADHERENCE: FailureCategory.CONSTRAINT_VIOLATED.value,
        Dimension.PLANNING_QUALITY: FailureCategory.OBJECTIVE_INCOMPLETE.value,
        Dimension.RECOVERY_QUALITY: FailureCategory.RECOVERY_FAILED.value,
        Dimension.USER_USEFULNESS: FailureCategory.OBJECTIVE_INCOMPLETE.value,
        Dimension.CONSISTENCY: FailureCategory.INTERNAL_CONTRADICTION.value,
    }.get(dimension, FailureCategory.NONE.value)


class LLMJudge:
    """Judge backed by a real model provider."""

    name = "llm-judge"

    def __init__(self, provider: ModelProvider, model: str = "", max_retries: int = 2) -> None:
        self.provider = provider
        self.model = model or getattr(provider, "model", "unknown")
        self.prompt_version = JUDGE_PROMPT_VERSION
        self.max_retries = max_retries

    def judge(self, scenario: Scenario, trace: SessionTrace) -> JudgeResponse:
        """Ask the model for one structured judgement.

        Raises:
            JudgeError: If no attempt produced parseable, evidence-bearing output.
        """
        request = ModelRequest(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            messages=[
                Message(
                    role="user",
                    content=(
                        f"{render_transcript(scenario, trace)}\n\n"
                        f"Score these dimensions: "
                        f"{', '.join(d.value for d in JUDGE_DIMENSIONS)}"
                    ),
                )
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        last_error = ""
        for attempt in range(self.max_retries + 1):
            response = self.provider.generate(request)
            try:
                payload = _extract_json(response.content)
                parsed = JudgeResponse.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("judge_output_unparseable", attempt=attempt, error=last_error)
                continue

            with_evidence = [verdict for verdict in parsed.scores if verdict.evidence]
            if not with_evidence:
                last_error = "no verdict carried trace evidence"
                logger.warning("judge_output_without_evidence", attempt=attempt)
                continue
            return JudgeResponse(scores=with_evidence)

        raise JudgeError(
            f"Judge produced unusable output after {self.max_retries + 1} attempts: {last_error}"
        )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Raises:
        ValueError: If no JSON object is present.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in judge response")
    return json.loads(text[start : end + 1])


class JudgeEvaluator(BaseEvaluator):
    """Runs a judge several times and aggregates the samples into results."""

    name = "llm_judge"
    kind = "judge"
    dimension = Dimension.TASK_COMPLETION

    def __init__(self, config: EvalForgeConfig, judge: Judge | None = None) -> None:
        """Construct with an explicit judge, defaulting to the deterministic mock."""
        super().__init__(config)
        self.judge_impl = judge or MockJudge(model=config.evaluation.judge_model)

    def evaluate(self, scenario: Scenario, trace: SessionTrace) -> list[EvaluationResult]:
        """Sample the judge, aggregate, and emit one result per dimension."""
        samples = max(1, self.config.evaluation.judge_samples)
        collected: dict[str, list[JudgeVerdict]] = {}

        for _ in range(samples):
            try:
                response = self.judge_impl.judge(scenario, trace)
            except JudgeError as exc:
                logger.warning("judge_sample_failed", error=str(exc))
                continue
            for verdict in response.scores:
                collected.setdefault(verdict.dimension, []).append(verdict)

        if not collected:
            return []

        results: list[EvaluationResult] = []
        for dimension_name, verdicts in sorted(collected.items()):
            try:
                dimension = Dimension(dimension_name)
            except ValueError:
                logger.warning("judge_unknown_dimension", dimension=dimension_name)
                continue

            score = self._aggregate([v.score for v in verdicts])
            confidence = statistics.median([v.confidence for v in verdicts])
            passed = sum(1 for v in verdicts if v.passed) > len(verdicts) / 2
            representative = min(verdicts, key=lambda v: abs(v.score - score))
            low_confidence = confidence < self.config.evaluation.low_confidence_threshold

            results.append(
                self.make_result(
                    scenario,
                    trace,
                    score=score,
                    passed=passed,
                    dimension=dimension,
                    suffix=f"judge::{dimension_name}",
                    evidence=[
                        Evidence(
                            kind="judge_evidence", excerpt=excerpt(item), detail="Cited by judge"
                        )
                        for item in representative.evidence[:2]
                    ],
                    reasoning=representative.reasoning_summary,
                    failure_category=_safe_category(representative.failure_category),
                    # Capped at MAJOR by construction: see ADR-004.
                    severity=Severity.MAJOR if not passed else Severity.INFO,
                    confidence=confidence,
                    metadata={
                        "judge_model": self.judge_impl.model,
                        "judge_prompt_version": self.judge_impl.prompt_version,
                        "samples": len(verdicts),
                        "aggregation": self.config.evaluation.judge_aggregation,
                        "sample_scores": [v.score for v in verdicts],
                        "score_spread": round(
                            max(v.score for v in verdicts) - min(v.score for v in verdicts), 4
                        ),
                        "low_confidence": low_confidence,
                    },
                )
            )
        return results

    def _aggregate(self, scores: list[float]) -> float:
        """Combine samples using the configured strategy."""
        mode = self.config.evaluation.judge_aggregation
        if mode == "mean":
            return statistics.fmean(scores)
        if mode == "majority":
            return 1.0 if sum(1 for s in scores if s >= 0.7) > len(scores) / 2 else 0.0
        return statistics.median(scores)


def _safe_category(value: str) -> FailureCategory:
    """Map a judge-supplied label onto the taxonomy, defaulting rather than raising."""
    try:
        return FailureCategory(value)
    except ValueError:
        return FailureCategory.NONE


def build_judge(config: EvalForgeConfig) -> Judge:
    """Construct the configured judge, falling back to the mock when offline.

    Falling back rather than raising is deliberate: a missing judge credential should
    degrade the report, not abort a run whose deterministic results are the ones that
    actually gate the release.
    """
    provider_name = config.evaluation.judge_provider.lower()
    if provider_name == "mock":
        return MockJudge(model=config.evaluation.judge_model)

    from evalforge.exceptions import ProviderUnavailableError
    from evalforge.providers.registry import build_provider

    try:
        provider = build_provider(config, provider_name)
    except ProviderUnavailableError as exc:
        logger.warning("judge_provider_unavailable", provider=provider_name, error=str(exc))
        return MockJudge(model=config.evaluation.judge_model)
    return LLMJudge(provider=provider, max_retries=config.evaluation.judge_max_retries)
