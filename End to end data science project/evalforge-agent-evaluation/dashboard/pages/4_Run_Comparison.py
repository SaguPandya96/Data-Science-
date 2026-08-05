"""Baseline-versus-candidate comparison and the regression gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

from evalforge.orchestration.comparison import compare_runs

st.set_page_config(page_title="Run comparison", page_icon="⚖️", layout="wide")


def main() -> None:
    """Render the run-comparison page."""
    st.title("Run comparison")
    st.caption(
        "Tolerances come from `configs/release_thresholds.yaml`. A breach makes "
        "`evalforge compare` exit non-zero, so a regression stops a pipeline."
    )

    options = data.run_options()
    if len(options) < 2:
        st.warning("At least two stored runs are needed. Run `evalforge demo`.")
        st.stop()

    names = list(options)
    baseline_label = st.sidebar.selectbox("Baseline", names, index=0, key="cmp_baseline")
    candidate_index = 1 if len(names) > 1 else 0
    candidate_label = st.sidebar.selectbox(
        "Candidate", names, index=candidate_index, key="cmp_candidate"
    )

    baseline_id = options[baseline_label]
    candidate_id = options[candidate_label]
    if baseline_id == candidate_id:
        st.info("Select two different runs to compare.")
        st.stop()

    config = data.get_config()
    baseline_metrics = data.metrics_of(baseline_id)
    candidate_metrics = data.metrics_of(candidate_id)

    report = compare_runs(
        baseline_metrics,
        candidate_metrics,
        config,
        data.sessions_of(baseline_id),
        data.sessions_of(candidate_id),
    )

    if report.gate_passed:
        st.success("**Regression gate passed** — no metric moved beyond its tolerance.")
    else:
        st.error(
            f"**Regression gate FAILED** — {len(report.regressions)} metric(s) beyond tolerance."
        )
        for item in report.regressions:
            st.markdown(f"- {item}")

    if "cliffs_delta_overall_score" in report.metadata:
        st.caption(
            f"Effect size on overall score (Cliff's delta): "
            f"{report.metadata['cliffs_delta_overall_score']:+.3f} "
            f"({report.metadata.get('cliffs_delta_interpretation')}). Non-parametric and "
            "ordinal, because session scores are bounded and heavily skewed."
        )

    st.divider()
    st.subheader("Metric deltas")

    frame = pd.DataFrame(
        [
            {
                "metric": delta.name,
                "baseline": delta.baseline,
                "candidate": delta.candidate,
                "change": delta.absolute_change,
                "tolerance": delta.tolerance,
                "verdict": "REGRESSED"
                if delta.regressed
                else ("ok" if delta.tolerance is not None else "not gated"),
            }
            for delta in report.deltas
        ]
    )

    def _highlight(row: pd.Series) -> list[str]:
        colour = "background-color: rgba(255,0,0,0.15)" if row["verdict"] == "REGRESSED" else ""
        return [colour] * len(row)

    st.dataframe(
        frame.style.apply(_highlight, axis=1).format(
            {
                "baseline": "{:.4f}",
                "candidate": "{:.4f}",
                "change": "{:+.4f}",
                "tolerance": "{:+.4f}",
            },
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
        height=560,
    )

    st.divider()
    tabs = st.tabs(["By category", "By difficulty", "By conversation length", "Critical failures"])

    with tabs[0]:
        _render_group(report.deltas_by_category, "category")
    with tabs[1]:
        _render_group(report.deltas_by_difficulty, "difficulty")
    with tabs[2]:
        _render_group(report.deltas_by_length, "turns")
    with tabs[3]:
        baseline_critical = int(baseline_metrics.get("critical_failures"))
        candidate_critical = int(candidate_metrics.get("critical_failures"))
        columns = st.columns(2)
        columns[0].metric("Baseline critical failures", baseline_critical)
        columns[1].metric(
            "Candidate critical failures",
            candidate_critical,
            delta=candidate_critical - baseline_critical,
            delta_color="inverse",
        )
        st.caption(
            "Critical failures are categorical. Any non-zero count forces a FAIL "
            "decision regardless of the weighted score (ADR-004)."
        )

        added = set(candidate_metrics.failure_counts) - set(baseline_metrics.failure_counts)
        if added:
            st.warning(f"Failure categories new in the candidate: {', '.join(sorted(added))}")

    st.divider()
    st.subheader("Cost and latency")
    columns = st.columns(4)
    columns[0].metric(
        "Avg latency (baseline)", f"{baseline_metrics.get('average_latency_ms'):.0f} ms"
    )
    latency_delta = candidate_metrics.get("average_latency_ms") - baseline_metrics.get(
        "average_latency_ms"
    )
    columns[1].metric(
        "Avg latency (candidate)",
        f"{candidate_metrics.get('average_latency_ms'):.0f} ms",
        delta=f"{latency_delta:+.0f} ms",
        delta_color="inverse",
    )
    columns[2].metric("Tokens (baseline)", f"{baseline_metrics.get('total_tokens'):,.0f}")
    columns[3].metric("Tokens (candidate)", f"{candidate_metrics.get('total_tokens'):,.0f}")
    st.caption(
        "Cost per *successful* task is the meaningful cost metric — a cheap agent that "
        "fails is not cheap. It reads zero here because the mock provider is free."
    )


def _render_group(grouped: dict[str, list], index_name: str) -> None:
    """Render one subgroup delta table."""
    if not grouped:
        st.info("No breakdown available.")
        return
    rows = []
    for group, deltas in grouped.items():
        row: dict[str, object] = {index_name: group}
        for delta in deltas:
            row[f"{delta.name}_baseline"] = delta.baseline
            row[f"{delta.name}_candidate"] = delta.candidate
            row[f"{delta.name}_change"] = delta.absolute_change
        rows.append(row)
    frame = pd.DataFrame(rows)
    st.dataframe(frame.style.format(precision=4), width="stretch", hide_index=True)


main()
