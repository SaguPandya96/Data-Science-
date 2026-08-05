"""EvalForge dashboard: Overview page.

Presentation only. Every number shown here is computed by ``evalforge.analytics`` and
is the same number the release report prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import data_access as data

st.set_page_config(page_title="EvalForge", page_icon="🔬", layout="wide")

DECISION_COLOUR = {"pass": "🟢", "conditional_pass": "🟡", "fail": "🔴"}


def main() -> None:
    """Render the overview page."""
    st.title("EvalForge")
    st.caption("Automated evaluation and adversarial stress testing for multi-turn AI agents")

    options = data.run_options()
    if not options:
        st.warning(
            "No evaluation runs are stored yet. Generate one with:\n\n```bash\nevalforge demo\n```"
        )
        st.stop()

    with st.sidebar:
        st.header("Run")
        label = st.selectbox("Select a run", list(options), key="overview_run")
        if st.button("Refresh data"):
            data.clear_caches()
            st.rerun()
    run_id = options[label]

    summary = data.summary_of(run_id)
    metrics = data.metrics_of(run_id)

    st.info(
        "**Deterministic demonstration results using simulated model behavior.** "
        "These figures characterise the evaluation system, not any language model's "
        "capability. See `docs/LIMITATIONS.md`."
        if summary.provider_name == "mock"
        else "Results from a live model provider."
    )

    decision = summary.release_decision.value
    st.subheader(
        f"{DECISION_COLOUR.get(decision, '⚪')} Release decision: "
        f"{decision.replace('_', ' ').upper()}"
    )

    top = st.columns(5)
    top[0].metric("Scenarios", metrics.session_count)
    pass_interval = metrics.intervals.get("pass_rate")
    top[1].metric(
        "Pass rate",
        f"{metrics.get('pass_rate'):.1%}",
        help=f"95% Wilson interval: {pass_interval.render()}" if pass_interval else None,
    )
    top[2].metric("Overall score", f"{metrics.get('overall_score'):.3f}")
    critical = int(metrics.get("critical_failures"))
    top[3].metric(
        "Critical failures",
        critical,
        delta=None if not critical else "blocks release",
        delta_color="inverse",
    )
    top[4].metric("Injection resistance", f"{metrics.get('prompt_injection_resistance'):.3f}")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Dimension scores")
        dimensions = {
            "Task completion": metrics.get("task_completion"),
            "Context retention": metrics.get("context_retention"),
            "Instruction adherence": metrics.get("instruction_adherence"),
            "Tool reliability": metrics.get("tool_reliability"),
            "Recovery quality": metrics.get("recovery_success"),
            "Consistency": metrics.get("consistency"),
            "Efficiency": metrics.get("efficiency"),
            "Safety": metrics.get("safety"),
        }
        st.bar_chart(dimensions, horizontal=True, height=320)

        st.subheader("Pass rate by conversation length")
        st.caption(
            "The headline multi-turn diagnostic: where context retention starts to fall "
            "shows up here as a curve rather than a single number."
        )
        length_frame = data.group_frame(metrics.by_length, "turns")
        if not length_frame.empty:
            length_frame["turns"] = length_frame["turns"].astype(int)
            length_frame = length_frame.sort_values("turns")
            st.line_chart(
                length_frame.set_index("turns")[
                    ["pass_rate", "context_retention", "instruction_adherence"]
                ],
                height=300,
            )

    with right:
        st.subheader("Run provenance")
        st.write(
            {
                "run_id": summary.run_id,
                "label": summary.label,
                "provider": summary.provider_name,
                "model": summary.model_name,
                "prompt_version": summary.prompt_version,
                "agent_version": summary.agent_version,
                "config_digest": summary.config_digest,
                "sessions": summary.session_count,
            }
        )
        st.caption(
            "The config digest ties this run to the exact rubric and thresholds that "
            "scored it (ADR-005)."
        )

        st.subheader("Uncertainty")
        for name, interval in metrics.intervals.items():
            st.text(f"{name}: {interval.render()}")
        st.caption(
            "Proportions use Wilson intervals; means use a seeded bootstrap. Intervals "
            "are suppressed below n=10."
        )

    st.divider()
    st.subheader("Scenario composition")
    category_frame = data.group_frame(metrics.by_category, "category")
    if not category_frame.empty:
        st.dataframe(
            category_frame[
                [
                    "category",
                    "n",
                    "pass_rate",
                    "overall_score",
                    "context_retention",
                    "instruction_adherence",
                    "critical_failures",
                ]
            ].style.format(
                {
                    "n": "{:.0f}",
                    "pass_rate": "{:.1%}",
                    "overall_score": "{:.3f}",
                    "context_retention": "{:.3f}",
                    "instruction_adherence": "{:.3f}",
                    "critical_failures": "{:.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "A passing decision means this agent revision cleared this suite under this "
        "configuration. It does not establish that the agent is safe or correct in general."
    )


main()
