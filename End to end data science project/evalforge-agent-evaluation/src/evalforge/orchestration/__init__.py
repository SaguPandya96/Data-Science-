"""Run orchestration: the pipeline, comparison and the regression gate."""

from __future__ import annotations

from evalforge.orchestration.comparison import (
    DEFAULT_COMPARED_METRICS,
    compare_runs,
    enforce_gate,
    render_comparison_table,
)
from evalforge.orchestration.pipeline import (
    RunResult,
    load_run_result,
    make_run_id,
    reevaluate_run,
    run_evaluation,
)

__all__ = [
    "DEFAULT_COMPARED_METRICS",
    "RunResult",
    "compare_runs",
    "enforce_gate",
    "load_run_result",
    "make_run_id",
    "reevaluate_run",
    "render_comparison_table",
    "run_evaluation",
]
