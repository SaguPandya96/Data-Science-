"""Tool reliability: selection, arguments, order, redundancy and recovery."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

st.set_page_config(page_title="Tool reliability", page_icon="🔧", layout="wide")


def main() -> None:
    """Render the tool-reliability page."""
    st.title("Tool reliability")
    st.caption(
        "Selection, arguments and order are scored separately because they fail "
        "differently and have different fixes."
    )

    options = data.run_options()
    if not options:
        st.warning("No runs stored. Run `evalforge demo` first.")
        st.stop()

    label = st.sidebar.selectbox("Select a run", list(options), key="tool_run")
    run_id = options[label]
    metrics = data.metrics_of(run_id)

    columns = st.columns(3)
    columns[0].metric("Selection accuracy", f"{metrics.get('tool_selection_accuracy'):.3f}")
    columns[1].metric("Argument exact match", f"{metrics.get('tool_argument_accuracy'):.3f}")
    columns[2].metric("Sequence accuracy", f"{metrics.get('tool_sequence_accuracy'):.3f}")

    st.caption(
        "Argument accuracy is exact match over the *pinned* argument subset only. "
        "Pinning every field would measure prompt-template stability, not correctness. "
        "Sequence accuracy is a longest-common-subsequence ratio, so a differently "
        "interleaved but dependency-respecting order is not counted as a defect."
    )

    st.divider()
    st.subheader("Reliability measures")
    reliability = metrics.tool_reliability
    if reliability:
        frame = pd.DataFrame({"measure": list(reliability), "score": list(reliability.values())})
        st.bar_chart(frame.set_index("measure")["score"], horizontal=True, height=320)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Recovery outcomes")
        st.metric("Recovery success rate", f"{metrics.get('recovery_success_rate'):.3f}")
        st.caption(
            "Scored on a rubric: retrying into a success is best, then adapting, then "
            "asking the user, then honestly reporting the gap. Inventing a result is "
            "not on this scale — it is a critical safety failure."
        )

        st.subheader("Efficiency")
        st.metric("Unnecessary call rate", f"{metrics.get('unnecessary_tool_call_rate'):.3f}")
        st.metric("Average latency", f"{metrics.get('average_latency_ms'):.0f} ms")
        st.metric("P95 latency", f"{metrics.get('p95_latency_ms'):.0f} ms")

    with right:
        st.subheader("Cascading errors")
        cascade = metrics.cascade
        if cascade:
            st.metric(
                "Sessions with corrupted input", f"{cascade.get('sessions_with_corruption', 0):.0f}"
            )
            st.metric("Mean propagation depth", f"{cascade.get('mean_propagation_depth', 0):.2f}")
            st.metric("Max propagation depth", f"{cascade.get('max_propagation_depth', 0):.0f}")
            st.metric("Mean downstream impact", f"{cascade.get('mean_downstream_impact', 0):.1%}")
            st.caption(
                "Depth matters more than count: an error that stops at the next step is "
                "a bug, one that reaches the executive summary is an incident."
            )

    st.divider()
    st.subheader("Results by injected failure type")
    st.caption(
        "How the agent handled each fault the harness injected. Faults are seeded, so "
        "these outcomes are reproducible across runs."
    )

    store = data.get_store()
    rows: list[dict[str, object]] = []
    counter: Counter[str] = Counter()
    recovered: Counter[str] = Counter()

    for session in data.sessions_of(run_id):
        try:
            trace = store.get_trace(run_id, session.session_id)
        except Exception:
            continue
        succeeded_tools = {c.tool_name for c in trace.all_tool_calls if c.succeeded}
        for call in trace.all_tool_calls:
            if call.injected_failure.value == "none":
                continue
            counter[call.injected_failure.value] += 1
            if call.succeeded or call.tool_name in succeeded_tools:
                recovered[call.injected_failure.value] += 1

    for fault, total in counter.most_common():
        rows.append(
            {
                "injected_failure": fault,
                "occurrences": total,
                "resolved": recovered[fault],
                "resolution_rate": recovered[fault] / total if total else 0.0,
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"occurrences": "{:.0f}", "resolved": "{:.0f}", "resolution_rate": "{:.1%}"}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No faults were injected in this run.")

    st.divider()
    st.subheader("Tool reliability by category")
    frame = data.group_frame(metrics.by_category, "category")
    if not frame.empty:
        st.bar_chart(frame.set_index("category")["tool_reliability"], horizontal=True, height=320)


main()
