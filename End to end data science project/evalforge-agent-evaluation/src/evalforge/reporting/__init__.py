"""Release-readiness assessment and report rendering.

Assessment (``release_readiness``) and presentation (``render``) are separate modules so
that how a decision is displayed can never change what the decision was.
"""

from __future__ import annotations

from evalforge.reporting.release_readiness import (
    LIVE_LABEL,
    SIMULATED_LABEL,
    build_report,
    decide_release,
    evaluate_thresholds,
)
from evalforge.reporting.render import (
    render_comparison_markdown,
    render_json,
    render_markdown,
    write_alignment,
    write_comparison,
    write_report,
)

__all__ = [
    "LIVE_LABEL",
    "SIMULATED_LABEL",
    "build_report",
    "decide_release",
    "evaluate_thresholds",
    "render_comparison_markdown",
    "render_json",
    "render_markdown",
    "write_alignment",
    "write_comparison",
    "write_report",
]
