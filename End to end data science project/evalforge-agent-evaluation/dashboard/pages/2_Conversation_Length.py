"""Conversation-length analysis: where multi-turn degradation begins."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

st.set_page_config(page_title="Conversation length", page_icon="📏", layout="wide")


def main() -> None:
    """Render the conversation-length page."""
    st.title("Conversation-length analysis")
    st.caption(
        "The experiment single-turn evaluation cannot run: hold the task constant and "
        "vary how long the agent has to hold onto it."
    )

    options = data.run_options()
    if not options:
        st.warning("No runs stored. Run `evalforge demo` first.")
        st.stop()

    label = st.sidebar.selectbox("Select a run", list(options), key="length_run")
    run_id = options[label]
    metrics = data.metrics_of(run_id)

    frame = data.group_frame(metrics.by_length, "turns")
    if frame.empty:
        st.warning("This run has no length breakdown.")
        st.stop()

    frame["turns"] = frame["turns"].astype(int)
    frame = frame.sort_values("turns").set_index("turns")

    st.subheader("Degradation curve")
    st.line_chart(
        frame[["pass_rate", "context_retention", "instruction_adherence", "overall_score"]],
        height=380,
    )

    columns = st.columns(len(frame))
    for column, (turns, row) in zip(columns, frame.iterrows(), strict=False):
        column.metric(
            f"{turns} turns",
            f"{row['pass_rate']:.0%}",
            help=f"n={int(row['n'])} sessions",
        )

    st.divider()
    st.subheader("Where retention starts to fall")

    retention = frame["context_retention"]
    best = retention.max()
    declines = [(turns, value) for turns, value in retention.items() if value < best - 0.02]
    if declines:
        onset, value = declines[0]
        st.warning(
            f"Context retention first drops materially at **{onset} turns** "
            f"({value:.3f} against a best of {best:.3f} at shorter lengths). "
            "Below that length the agent holds context; above it, it does not."
        )
    else:
        st.success(
            "Context retention is flat across every length in this suite. This agent "
            "revision shows no measurable degradation with conversation length."
        )

    st.caption(
        "Read with the sample sizes below. A drop measured over a handful of sessions "
        "is noise; the n column is shown for exactly this reason."
    )

    st.divider()
    st.subheader("Per-length detail")
    st.dataframe(
        frame.reset_index().style.format(
            {
                "turns": "{:.0f}",
                "n": "{:.0f}",
                "pass_rate": "{:.1%}",
                "overall_score": "{:.3f}",
                "context_retention": "{:.3f}",
                "instruction_adherence": "{:.3f}",
                "tool_reliability": "{:.3f}",
                "task_completion": "{:.3f}",
                "recovery": "{:.3f}",
                "safety": "{:.3f}",
                "critical_failures": "{:.0f}",
                "mean_latency_ms": "{:.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Session scores by length")
    sessions = data.sessions_frame(run_id)
    if not sessions.empty:
        st.scatter_chart(sessions, x="turns", y="score", color="category", height=380)
        st.caption(
            "Each point is one session. Vertical spread within a length shows how much "
            "of the variation is scenario difficulty rather than length itself."
        )


main()
