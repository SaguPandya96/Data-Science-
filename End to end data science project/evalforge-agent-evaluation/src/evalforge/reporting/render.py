"""Report rendering.

Markdown for humans, JSON for machines, from the same
:class:`~evalforge.schemas.evaluation.ReleaseReadinessReport`. Rendering is deliberately
separated from the assessment in ``release_readiness.py``: how a decision is presented
should never be able to change what the decision was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from evalforge.config import EvalForgeConfig
from evalforge.logging_config import get_logger
from evalforge.schemas.evaluation import ComparisonReport, ReleaseReadinessReport

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Metrics shown in the report's headline table, in reading order.
HEADLINE_METRICS: tuple[str, ...] = (
    "pass_rate",
    "overall_score",
    "task_completion",
    "context_retention",
    "instruction_adherence",
    "tool_reliability",
    "tool_selection_accuracy",
    "tool_argument_accuracy",
    "tool_sequence_accuracy",
    "recovery_success_rate",
    "prompt_injection_resistance",
    "goal_drift_rate",
    "unsupported_claim_rate",
    "contradiction_rate",
    "unnecessary_tool_call_rate",
    "error_propagation_depth",
    "downstream_impact_score",
    "average_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "total_tokens",
    "estimated_cost",
    "estimated_cost_per_success",
    "critical_failures",
)


def _environment() -> Environment:
    """Jinja environment with strict undefined.

    ``StrictUndefined`` is chosen so a template referencing a field that no longer
    exists fails loudly at render time instead of silently emitting a blank cell in a
    release report.
    """
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def render_markdown(report: ReleaseReadinessReport, config: EvalForgeConfig) -> str:
    """Render the release report as Markdown."""
    metrics = report.metadata.get("metrics", {})
    scalars: dict[str, float] = metrics.get("scalars", {}) if isinstance(metrics, dict) else {}
    raw_intervals = metrics.get("intervals", {}) if isinstance(metrics, dict) else {}

    headline = {name: scalars[name] for name in HEADLINE_METRICS if name in scalars}
    intervals = {
        name: _IntervalView(payload)
        for name, payload in raw_intervals.items()
        if isinstance(payload, dict)
    }

    length_rows = sorted(
        report.metrics_by_length.items(),
        key=lambda item: int(item[0]) if item[0].isdigit() else 0,
    )

    template = _environment().get_template("release_report.md.j2")
    return template.render(
        report=report,
        config=config,
        headline_metrics=headline,
        intervals=intervals,
        length_rows=length_rows,
    )


class _IntervalView:
    """Attribute access over a serialised interval, for template convenience."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.point = float(payload.get("point", 0.0))
        self.lower = payload.get("lower")
        self.upper = payload.get("upper")
        self.n = int(payload.get("n", 0))
        self.method = str(payload.get("method", ""))


def render_json(report: ReleaseReadinessReport) -> str:
    """Render the release report as indented JSON."""
    return report.model_dump_json(indent=2)


def write_report(
    report: ReleaseReadinessReport,
    config: EvalForgeConfig,
    directory: Path,
    stem: str = "release_readiness",
) -> tuple[Path, Path]:
    """Write both report formats.

    Returns:
        The Markdown and JSON paths, in that order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"

    markdown_path.write_text(render_markdown(report, config), encoding="utf-8")
    json_path.write_text(render_json(report), encoding="utf-8")

    logger.info("report_written", markdown=str(markdown_path), json=str(json_path))
    return markdown_path, json_path


def render_comparison_markdown(comparison: ComparisonReport) -> str:
    """Render a baseline-versus-candidate comparison as Markdown."""
    lines = [
        "# EvalForge Regression Comparison",
        "",
        f"- **Baseline:** `{comparison.baseline_run_id}`",
        f"- **Candidate:** `{comparison.candidate_run_id}`",
        f"- **Generated:** {comparison.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Gate:** {'PASSED' if comparison.gate_passed else 'FAILED'}",
        "",
    ]

    if "cliffs_delta_overall_score" in comparison.metadata:
        lines.extend(
            [
                f"Effect size on overall score (Cliff's delta): "
                f"{comparison.metadata['cliffs_delta_overall_score']} "
                f"({comparison.metadata.get('cliffs_delta_interpretation', 'n/a')}). "
                "Non-parametric and ordinal, because 0-1 session scores are bounded and "
                "heavily skewed.",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric deltas",
            "",
            "| Metric | Baseline | Candidate | Change | Tolerance | Verdict |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for delta in comparison.deltas:
        tolerance = "-" if delta.tolerance is None else f"{delta.tolerance:+.4f}"
        if delta.regressed:
            verdict = "**REGRESSED**"
        elif delta.tolerance is None:
            verdict = "not gated"
        else:
            verdict = "ok"
        lines.append(
            f"| {delta.name.replace('_', ' ')} | {delta.baseline:.4f} | "
            f"{delta.candidate:.4f} | {delta.absolute_change:+.4f} | {tolerance} | {verdict} |"
        )

    if comparison.regressions:
        lines.extend(["", "## Regressions beyond tolerance", ""])
        lines.extend(f"- {item}" for item in comparison.regressions)

    if comparison.improvements:
        lines.extend(["", "## Improvements", ""])
        lines.extend(f"- {item}" for item in comparison.improvements)

    if comparison.deltas_by_length:
        lines.extend(
            [
                "",
                "## Pass rate by conversation length",
                "",
                "| Turns | Baseline | Candidate | Change |",
                "|---:|---:|---:|---:|",
            ]
        )
        for bucket, deltas in sorted(
            comparison.deltas_by_length.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else 0,
        ):
            pass_delta = next((d for d in deltas if d.name == "pass_rate"), None)
            if pass_delta:
                lines.append(
                    f"| {bucket} | {pass_delta.baseline:.4f} | {pass_delta.candidate:.4f} | "
                    f"{pass_delta.absolute_change:+.4f} |"
                )

    lines.extend(
        [
            "",
            "---",
            "",
            "*Tolerances are configured in `configs/release_thresholds.yaml`. A breach "
            "makes `evalforge compare` exit non-zero so a regression stops a pipeline "
            "rather than appearing in a report nobody reads.*",
        ]
    )
    return "\n".join(lines)


def write_comparison(
    comparison: ComparisonReport,
    directory: Path,
    stem: str = "regression_comparison",
) -> tuple[Path, Path]:
    """Write the comparison in both formats.

    Returns:
        The Markdown and JSON paths, in that order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    json_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    return markdown_path, json_path


def write_alignment(report: Any, directory: Path, stem: str = "evaluator_alignment") -> Path:
    """Write the alignment analysis as JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.json"
    payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
