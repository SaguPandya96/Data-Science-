"""Failure analysis: what broke, how badly, and what it looked like."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

st.set_page_config(page_title="Failure analysis", page_icon="🐞", layout="wide")

SEVERITY_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡", "info": "⚪"}


def main() -> None:
    """Render the failure-analysis page."""
    st.title("Failure analysis")

    options = data.run_options()
    if not options:
        st.warning("No runs stored. Run `evalforge demo` first.")
        st.stop()

    label = st.sidebar.selectbox("Select a run", list(options), key="failure_run")
    run_id = options[label]
    metrics = data.metrics_of(run_id)

    left, right = st.columns(2)

    with left:
        st.subheader("Failures by category")
        frame = data.failure_frame(run_id)
        if frame.empty:
            st.success("No failures recorded in this run.")
        else:
            st.bar_chart(
                frame.set_index("failure_category")["sessions"],
                horizontal=True,
                height=max(300, 26 * len(frame)),
            )

    with right:
        st.subheader("Failures by severity")
        severity = data.severity_frame(run_id)
        if severity.empty:
            st.success("No failing checks.")
        else:
            st.bar_chart(severity.set_index("severity")["count"], height=300)
            st.caption(
                "A single **critical** failure blocks release regardless of the weighted "
                "score (ADR-004). Some failures are categorical, not statistical."
            )

    st.divider()

    st.subheader("Most frequent root causes")
    st.caption(
        "Ordered by how many sessions each affected. This is the list the release "
        "report's remediation section is built from."
    )
    frame = data.failure_frame(run_id)
    if not frame.empty:
        total = metrics.session_count or 1
        display = frame.copy()
        display["share_of_sessions"] = display["sessions"] / total
        st.dataframe(
            display.style.format({"sessions": "{:.0f}", "share_of_sessions": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Representative failed scenarios")
    st.caption(
        "The most severe example of each category, with the trace evidence that "
        "justified the verdict. Every automated failure points at something in the trace."
    )

    examples = data.representative_failures(run_id)
    if not examples:
        st.success("Nothing to show — no failing checks in this run.")
        return

    categories = sorted(examples)
    chosen = st.selectbox("Failure category", categories, key="failure_category")

    for item in examples[chosen]:
        icon = SEVERITY_ICON.get(item["severity"], "⚪")
        with st.expander(
            f"{icon} {item['scenario_id']} · turn {item['turn_index']} · "
            f"{item['evaluator']} · score {item['score']:.2f}",
            expanded=False,
        ):
            st.markdown(f"**Verdict:** {item['reasoning']}")
            for evidence in item["evidence"]:
                st.markdown(f"**Evidence — {evidence['kind']}**")
                st.code(evidence["excerpt"] or "(empty)", language=None)
                if evidence["detail"]:
                    st.caption(evidence["detail"])
            st.caption(f"Session `{item['session_id']}`")

    st.divider()
    st.subheader("Critical failures")
    sessions = [item for item in data.sessions_of(run_id) if item.critical_failures]
    if not sessions:
        st.success("No release-blocking failures in this run.")
        return

    st.error(
        f"{len(sessions)} session(s) recorded a release-blocking failure. Any one of "
        "these forces a FAIL decision."
    )
    for item in sessions[:25]:
        st.markdown(
            f"- `{item.scenario_id}` ({item.scenario_category}, {item.turn_count} turns, "
            f"score {item.overall_score:.3f}) — {', '.join(item.critical_failures)}"
        )


main()
