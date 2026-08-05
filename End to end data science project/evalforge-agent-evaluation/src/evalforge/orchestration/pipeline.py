"""The evaluation pipeline: scenarios in, a scored and persisted run out.

One entry point, :func:`run_evaluation`, wires the whole system together. Keeping it in
one place means the CLI, the demo command, the tests and CI all execute *the same*
pipeline — a property the regression gate depends on, since a candidate and a baseline
must differ only in the agent, never in how they were run.

The trace is persisted before its index row is written, so a crash leaves an orphaned
trace file rather than an index pointing at nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from evalforge.agents.productivity_agent import ProductivityAgent
from evalforge.analytics.metrics import RunMetrics, compute_metrics
from evalforge.config import EvalForgeConfig
from evalforge.evaluators.aggregation import aggregate_session
from evalforge.evaluators.registry import build_evaluators, evaluate_session
from evalforge.ids import stable_id
from evalforge.logging_config import bind_run_context, clear_run_context, get_logger
from evalforge.providers.registry import build_provider
from evalforge.reporting.release_readiness import decide_release, evaluate_thresholds
from evalforge.schemas.evaluation import EvaluationResult, RunSummary, SessionSummary
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace
from evalforge.storage.store import RunStore
from evalforge.tools.registry import ToolRegistry

logger = get_logger(__name__)

#: Callback signature for progress reporting, ``(completed, total, scenario_id)``.
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class RunResult:
    """Everything one pipeline execution produced."""

    run_id: str
    summary: RunSummary
    session_summaries: list[SessionSummary] = field(default_factory=list)
    traces: list[SessionTrace] = field(default_factory=list)
    results: list[EvaluationResult] = field(default_factory=list)
    # Always populated by every constructor here. A default is supplied only so the
    # field can follow the list fields; callers never see an unset one.
    metrics: RunMetrics = field(default_factory=lambda: RunMetrics(run_id=""))

    @property
    def pass_rate(self) -> float:
        """Fraction of sessions that passed."""
        if not self.session_summaries:
            return 0.0
        return sum(1 for item in self.session_summaries if item.passed) / len(
            self.session_summaries
        )

    @property
    def critical_failure_count(self) -> int:
        """Total release-blocking failures across the run."""
        return sum(len(item.critical_failures) for item in self.session_summaries)


def make_run_id(label: str, seed: int, suite_id: str, profile: str) -> str:
    """Derive a deterministic run id.

    Deterministic rather than random so regenerating the demonstration overwrites the
    same run instead of accumulating near-duplicates, and so the committed artifacts
    keep stable paths across regeneration.
    """
    return stable_id("run", label, seed, suite_id, profile)


def _aborted_session(
    run_id: str, scenario: Scenario, error: Exception, store: RunStore
) -> SessionSummary:
    """Record a scenario that could not be executed as a failed session.

    Scored zero rather than omitted. An absent session would quietly shrink the
    denominator and inflate every rate in the report, which is the wrong direction for
    an evaluation system to be wrong in.
    """
    summary = SessionSummary(
        run_id=run_id,
        session_id=stable_id("ses", run_id, scenario.scenario_id),
        scenario_id=scenario.scenario_id,
        scenario_category=scenario.category.value,
        scenario_difficulty=scenario.difficulty.value,
        turn_count=scenario.turn_count,
        overall_score=0.0,
        passed=False,
        metadata={
            "aborted": True,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        },
    )
    store.save_session(summary)
    return summary


def run_evaluation(
    scenarios: list[Scenario],
    config: EvalForgeConfig,
    store: RunStore,
    label: str = "baseline",
    profile: str = "baseline",
    provider_name: str | None = None,
    seed: int = 42,
    suite_id: str = "core",
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Execute and score a scenario suite end to end.

    Args:
        scenarios: The suite to run.
        config: Effective configuration.
        store: Where runs, traces and evaluations are persisted.
        label: Human-readable run label, e.g. ``baseline`` or ``candidate``.
        profile: Mock behaviour profile. Ignored by external providers.
        provider_name: Override for the configured provider.
        seed: Master seed for fault injection and degradation decisions.
        suite_id: Identifier of the suite being executed.
        run_id: Explicit run id, otherwise derived deterministically.
        progress: Optional callback invoked after each scenario.

    Returns:
        A :class:`RunResult` with summaries, traces, results and metrics.
    """
    resolved_run_id = run_id or make_run_id(label, seed, suite_id, profile)
    bind_run_context(resolved_run_id)
    started = datetime.now(UTC)

    provider = build_provider(config, provider_name, profile)
    registry = ToolRegistry()
    agent = ProductivityAgent(provider, registry, config, resolved_run_id, run_seed=seed)
    evaluators = build_evaluators(config)

    store.save_scenarios(resolved_run_id, scenarios)

    session_summaries: list[SessionSummary] = []
    traces: list[SessionTrace] = []
    all_results: list[EvaluationResult] = []

    logger.info(
        "run_started",
        label=label,
        profile=profile,
        provider=getattr(provider, "name", "unknown"),
        scenarios=len(scenarios),
        seed=seed,
    )

    for index, scenario in enumerate(scenarios, start=1):
        try:
            outcome = agent.run(scenario)
        except Exception as exc:
            # One scenario must never take down a run. A real model can exceed its
            # context window on a long conversation, or return something a tool cannot
            # digest, and losing every other completed session to that is a far worse
            # outcome than recording this one as failed. The session is scored zero and
            # the reason is preserved, so an aborted scenario is visible in the report
            # rather than silently absent from it.
            logger.warning(
                "scenario_aborted",
                scenario_id=scenario.scenario_id,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            session_summaries.append(_aborted_session(resolved_run_id, scenario, exc, store))
            if progress:
                progress(index, len(scenarios), scenario.scenario_id)
            continue

        trace = outcome.trace

        # Trace first, index second: an orphaned trace is recoverable, an index row
        # pointing at a missing trace is not.
        store.save_trace(trace)

        results = evaluate_session(scenario, trace, evaluators)
        summary = aggregate_session(scenario, trace, results, config)
        summary.metadata["first_failure_turn"] = _first_failure_turn(results)

        store.save_evaluations(results)
        store.save_session(summary)

        traces.append(trace)
        all_results.extend(results)
        session_summaries.append(summary)

        if progress:
            progress(index, len(scenarios), scenario.scenario_id)

    metrics = compute_metrics(resolved_run_id, session_summaries, all_results, seed=seed)

    # Decide the release verdict here rather than defaulting it. Storing a placeholder
    # and only correcting it when `evalforge report` happens to run left every freshly
    # completed run recorded as FAIL, so anything reading the store directly (the
    # dashboard, `evalforge runs`) reported a passing agent as failing. The report
    # command calls the same two functions, so the two can no longer disagree.
    threshold_checks = evaluate_thresholds(metrics, config)
    decision, _ = decide_release(threshold_checks, int(metrics.get("critical_failures")))

    run_summary = RunSummary(
        run_id=resolved_run_id,
        label=label,
        suite_id=suite_id,
        provider_name=getattr(provider, "name", "mock"),
        model_name=getattr(provider, "model", "mock"),
        prompt_version=config.agent.prompt_version,
        agent_version=config.agent.version,
        started_at=started,
        completed_at=datetime.now(UTC),
        session_count=len(session_summaries),
        metrics=dict(metrics.scalars),
        release_decision=decision,
        config_digest=config.digest,
        metadata={
            "profile": profile,
            "seed": seed,
            "scenario_count": len(scenarios),
            "evaluator_count": len(evaluators),
        },
    )
    store.save_run(run_summary)

    logger.info(
        "run_complete",
        run_id=resolved_run_id,
        sessions=len(session_summaries),
        pass_rate=round(metrics.get("pass_rate"), 4),
        critical_failures=int(metrics.get("critical_failures")),
    )
    clear_run_context()

    return RunResult(
        run_id=resolved_run_id,
        summary=run_summary,
        session_summaries=session_summaries,
        traces=traces,
        results=all_results,
        metrics=metrics,
    )


def reevaluate_run(
    run_id: str,
    config: EvalForgeConfig,
    store: RunStore,
) -> RunResult:
    """Re-score a stored run without re-executing the agent.

    This is the payoff of making the trace the single source of truth: an evaluator fix
    can be applied to historical runs, and two runs scored months apart can be compared
    under one rubric.
    """
    stored = store.get_run(run_id)
    scenarios = {s.scenario_id: s for s in store.get_scenarios(run_id)}
    evaluators = build_evaluators(config)

    session_summaries: list[SessionSummary] = []
    all_results: list[EvaluationResult] = []
    traces: list[SessionTrace] = []

    for trace in store.iter_traces(run_id):
        scenario = scenarios.get(trace.scenario_id)
        if scenario is None:
            logger.warning("scenario_missing_for_trace", scenario_id=trace.scenario_id)
            continue
        results = evaluate_session(scenario, trace, evaluators)
        summary = aggregate_session(scenario, trace, results, config)
        summary.metadata["first_failure_turn"] = _first_failure_turn(results)
        store.save_evaluations(results)
        store.save_session(summary)
        traces.append(trace)
        all_results.extend(results)
        session_summaries.append(summary)

    metrics = compute_metrics(run_id, session_summaries, all_results)
    stored.metrics = dict(metrics.scalars)
    stored.config_digest = config.digest
    store.save_run(stored)

    return RunResult(
        run_id=run_id,
        summary=stored,
        session_summaries=session_summaries,
        traces=traces,
        results=all_results,
        metrics=metrics,
    )


def load_run_result(run_id: str, store: RunStore) -> RunResult:
    """Rebuild a :class:`RunResult` from persisted records.

    Traces are not loaded: the dashboard and reports need summaries and results, and
    reading 150 full traces to render a metrics page would be wasteful. Callers that
    need a trace fetch it individually via :meth:`RunStore.get_trace`.
    """
    summary = store.get_run(run_id)
    summaries = store.get_sessions(run_id)
    results = store.get_evaluations(run_id)
    metrics = compute_metrics(run_id, summaries, results)
    return RunResult(
        run_id=run_id,
        summary=summary,
        session_summaries=summaries,
        traces=[],
        results=results,
        metrics=metrics,
    )


def _first_failure_turn(results: list[EvaluationResult]) -> int | None:
    """Earliest turn carrying a failing turn-level result, for position-bias analysis."""
    turns = [
        result.turn_index
        for result in results
        if not result.passed and result.turn_index is not None
    ]
    return min(turns) if turns else None
