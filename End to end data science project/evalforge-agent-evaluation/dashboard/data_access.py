"""Thin read layer between Streamlit and EvalForge.

Every function here loads or reshapes data. **No metric is computed in this module and
none is computed in a page** — the dashboard calls into ``evalforge.analytics`` for
that. The rule exists so that a number shown on screen is the same number the release
report prints, computed by the same unit-tested code, rather than a second
implementation that quietly drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make the package importable from a bare checkout. Locally EvalForge is pip-installed,
# but a hosted deploy (Streamlit Community Cloud) only clones the repository and installs
# requirements.txt, so `src/` has to be put on the path before the imports below.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evalforge.analytics.alignment import build_alignment_report
from evalforge.analytics.metrics import RunMetrics, compute_metrics, failure_examples
from evalforge.config import EvalForgeConfig, load_config
from evalforge.schemas.annotation import AlignmentReport, HumanAnnotation
from evalforge.schemas.evaluation import EvaluationResult, RunSummary, SessionSummary
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace
from evalforge.storage.store import RunStore


@st.cache_resource
def get_config() -> EvalForgeConfig:
    """Load the effective configuration once per session."""
    return load_config()


@st.cache_resource
def get_store() -> RunStore:
    """Open the run store once per session."""
    config = get_config()
    return RunStore(config.paths.runs, config.storage.database_name)


#: Scenarios generated when bootstrapping an empty store. Small on purpose: a hosted
#: demo has to finish inside a page load, and the point is to show the pipeline working,
#: not to reproduce the 150-scenario figures quoted in the README.
BOOTSTRAP_SCENARIOS = 24


@st.cache_resource(show_spinner=False)
def ensure_demo_data() -> bool:
    """Generate a small baseline/candidate pair if the store is empty.

    Run data is regenerable from a seed and therefore deliberately not committed, which
    leaves a fresh clone with nothing to display. Rather than ship a fossilised database,
    the app builds its own on first load. Cached as a resource so it happens once per
    server process, not once per page view.

    Returns:
        True if data was generated, False if the store already had runs.
    """
    store = get_store()
    if store.list_runs():
        return False

    from evalforge.orchestration.pipeline import run_evaluation
    from evalforge.scenarios.generator import generate_scenarios

    config = get_config()
    with st.spinner(
        f"First load: generating {BOOTSTRAP_SCENARIOS} adversarial scenarios and "
        "evaluating two agent revisions. This runs once and takes about a minute."
    ):
        scenarios = generate_scenarios(count=BOOTSTRAP_SCENARIOS, seed=42, config=config)
        for label, profile in (("baseline", "baseline"), ("candidate", "candidate")):
            run_evaluation(scenarios, config, store, label=label, profile=profile)
    return True


@st.cache_data(ttl=30)
def list_runs() -> list[dict[str, Any]]:
    """Every stored run as plain dicts, so Streamlit can cache them."""
    return [item.model_dump(mode="json") for item in get_store().list_runs()]


def run_options() -> dict[str, str]:
    """``{display_label: run_id}`` for run pickers.

    Bootstraps demo data first, so landing directly on any page works rather than only
    the overview. Every page reaches the store through here.
    """
    if ensure_demo_data():
        list_runs.clear()

    options: dict[str, str] = {}
    for item in list_runs():
        label = (
            f"{item['label'] or 'run'} · {item['run_id']} · "
            f"{item['session_count']} sessions · {item['release_decision']}"
        )
        options[label] = item["run_id"]
    return options


@st.cache_data(ttl=30)
def load_run(run_id: str) -> dict[str, Any]:
    """Load one run's summary, sessions, results and metrics.

    Returns plain dicts and a :class:`RunMetrics` so Streamlit's cache can hold them.
    """
    store = get_store()
    summary = store.get_run(run_id)
    sessions = store.get_sessions(run_id)
    results = store.get_evaluations(run_id)
    metrics = compute_metrics(run_id, sessions, results)
    return {
        "summary": summary,
        "sessions": sessions,
        "results": results,
        "metrics": metrics,
    }


def summary_of(run_id: str) -> RunSummary:
    """Run-level summary."""
    return load_run(run_id)["summary"]


def sessions_of(run_id: str) -> list[SessionSummary]:
    """Session summaries for a run."""
    return load_run(run_id)["sessions"]


def results_of(run_id: str) -> list[EvaluationResult]:
    """Evaluation results for a run."""
    return load_run(run_id)["results"]


def metrics_of(run_id: str) -> RunMetrics:
    """Computed metrics for a run."""
    return load_run(run_id)["metrics"]


def sessions_frame(run_id: str) -> pd.DataFrame:
    """Session summaries as a DataFrame, for tables and charts."""
    sessions = sessions_of(run_id)
    if not sessions:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "session_id": item.session_id,
                "scenario_id": item.scenario_id,
                "category": item.scenario_category,
                "difficulty": item.scenario_difficulty,
                "turns": item.turn_count,
                "score": item.overall_score,
                "passed": item.passed,
                "task_completion": item.task_completion_score,
                "context_retention": item.context_retention_score,
                "instruction_adherence": item.instruction_adherence_score,
                "tool_reliability": item.tool_reliability_score,
                "recovery": item.recovery_score,
                "consistency": item.consistency_score,
                "efficiency": item.efficiency_score,
                "safety": item.safety_score,
                "latency_ms": item.total_latency,
                "tokens": item.total_tokens,
                "critical_failures": len(item.critical_failures),
                "failures": ", ".join(item.failure_categories),
            }
            for item in sessions
        ]
    )


def group_frame(grouped: dict[str, dict[str, float]], index_name: str) -> pd.DataFrame:
    """Turn a metrics breakdown into a DataFrame."""
    if not grouped:
        return pd.DataFrame()
    frame = pd.DataFrame(grouped).T
    frame.index.name = index_name
    return frame.reset_index()


def failure_frame(run_id: str) -> pd.DataFrame:
    """Failure counts for a run."""
    counts = metrics_of(run_id).failure_counts
    if not counts:
        return pd.DataFrame()
    return pd.DataFrame(
        {"failure_category": list(counts), "sessions": list(counts.values())}
    ).sort_values("sessions", ascending=False)


def severity_frame(run_id: str) -> pd.DataFrame:
    """Failing results grouped by severity."""
    rows: dict[str, int] = {}
    for item in results_of(run_id):
        if item.passed:
            continue
        rows[item.severity.value] = rows.get(item.severity.value, 0) + 1
    if not rows:
        return pd.DataFrame()
    order = ["critical", "major", "minor", "info"]
    return pd.DataFrame(
        {
            "severity": [s for s in order if s in rows],
            "count": [rows[s] for s in order if s in rows],
        }
    )


def representative_failures(run_id: str) -> dict[str, list[dict[str, Any]]]:
    """Representative failing results per category."""
    return failure_examples(results_of(run_id))


@st.cache_data(ttl=30)
def load_trace(run_id: str, session_id: str) -> SessionTrace:
    """Load one session trace."""
    return get_store().get_trace(run_id, session_id)


@st.cache_data(ttl=30)
def load_scenario(run_id: str, scenario_id: str) -> Scenario | None:
    """Load one scenario definition."""
    return get_store().get_scenario(run_id, scenario_id)


def evaluations_for_session(run_id: str, session_id: str) -> list[EvaluationResult]:
    """Every evaluation result for one session."""
    return get_store().get_evaluations(run_id, session_id)


def annotations_for(run_id: str, blind_only: bool = True) -> list[HumanAnnotation]:
    """Stored annotations for a run."""
    return get_store().get_annotations(run_id, blind_only=blind_only)


def alignment_for(run_id: str) -> AlignmentReport | None:
    """Compute the alignment analysis for a run, or ``None`` with no annotations."""
    annotations = annotations_for(run_id)
    if not annotations:
        return None
    sessions = sessions_of(run_id)
    verbosity = {
        item.session_id: sum(len(turn.assistant_message) for turn in trace.turns)
        for item in sessions
        if (trace := _safe_trace(run_id, item.session_id)) is not None
    }
    return build_alignment_report(run_id, annotations, sessions, verbosity)


def _safe_trace(run_id: str, session_id: str) -> SessionTrace | None:
    """Load a trace, returning ``None`` when it is missing rather than raising."""
    try:
        return load_trace(run_id, session_id)
    except Exception:
        return None


def synthetic_annotation_count(run_id: str) -> int:
    """How many stored annotations are synthetic rather than human-authored.

    Surfaced prominently wherever agreement statistics are shown. Statistics computed
    over simulated labels describe the simulator, not human judgement, and presenting
    them without that caveat would be misleading.
    """
    return sum(
        1
        for item in annotations_for(run_id, blind_only=False)
        if item.metadata.get("synthetic") is True
    )


def clear_caches() -> None:
    """Drop cached data so a freshly written run appears."""
    st.cache_data.clear()
