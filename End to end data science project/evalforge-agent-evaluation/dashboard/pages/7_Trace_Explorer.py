"""Trace explorer: one session, end to end, with every verdict and its evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

st.set_page_config(page_title="Trace explorer", page_icon="🔍", layout="wide")

SEVERITY_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡", "info": "⚪"}


def main() -> None:
    """Render the trace explorer."""
    st.title("Trace explorer")
    st.caption(
        "Everything EvalForge concludes about an agent is derived from the trace. This "
        "page shows the whole of it for one session, alongside every verdict."
    )

    options = data.run_options()
    if not options:
        st.warning("No runs stored. Run `evalforge demo` first.")
        st.stop()

    with st.sidebar:
        label = st.selectbox("Run", list(options), key="trace_run")
        run_id = options[label]
        sessions = data.sessions_of(run_id)
        only_failures = st.checkbox("Failing sessions only", value=True)
        pool = [item for item in sessions if not item.passed] if only_failures else sessions
        if not pool:
            pool = sessions
        categories = sorted({item.scenario_category for item in pool})
        category = st.selectbox("Category", ["(all)", *categories])
        if category != "(all)":
            pool = [item for item in pool if item.scenario_category == category]

        if not pool:
            st.warning("No sessions match this filter.")
            st.stop()

        chosen = st.selectbox(
            "Session",
            pool,
            format_func=lambda item: (
                f"{item.scenario_id[:18]} · {item.turn_count}t · "
                f"{item.overall_score:.2f} · {'pass' if item.passed else 'FAIL'}"
            ),
        )

    session = chosen
    scenario = data.load_scenario(run_id, session.scenario_id)
    try:
        trace = data.load_trace(run_id, session.session_id)
    except Exception as exc:
        st.error(f"Could not load this trace: {exc}")
        st.stop()

    results = data.evaluations_for_session(run_id, session.session_id)

    st.subheader(scenario.name if scenario else session.scenario_id)
    columns = st.columns(6)
    columns[0].metric("Score", f"{session.overall_score:.3f}")
    columns[1].metric("Verdict", "PASS" if session.passed else "FAIL")
    columns[2].metric("Turns", session.turn_count)
    columns[3].metric("Critical", len(session.critical_failures))
    columns[4].metric("Tool calls", len(trace.all_tool_calls))
    columns[5].metric("Latency", f"{session.total_latency:.0f} ms")

    if session.critical_failures:
        st.error("**Release-blocking failure(s):** " + "; ".join(session.critical_failures))

    if scenario:
        with st.expander("Scenario contract — what the agent was expected to do", expanded=False):
            st.markdown(f"**{scenario.description}**")
            st.write("**Active constraints**")
            for constraint in scenario.active_constraints:
                st.markdown(
                    f"- `{constraint.constraint_id}` {constraint.description} "
                    f"(from turn {constraint.turn_added}"
                    + (f", revoked at {constraint.turn_removed}" if constraint.turn_removed else "")
                    + ")"
                )
            st.write("**Expected end-state facts**")
            for key, fact in sorted(scenario.final_fact_values().items()):
                st.markdown(f"- `{key}` = **{fact.value}**{' (critical)' if fact.critical else ''}")
            st.write("**Expected tool sequence**")
            st.code(" → ".join(tool.value for tool in scenario.expected_tool_sequence) or "(none)")
            st.write("**Failure conditions**")
            for condition in scenario.failure_conditions:
                st.markdown(f"- {condition}")

    st.divider()

    results_by_turn: dict[int, list] = {}
    session_level = []
    for item in results:
        if item.turn_index is None:
            session_level.append(item)
        else:
            results_by_turn.setdefault(item.turn_index, []).append(item)

    st.subheader("Conversation and agent state")

    for turn in trace.turns:
        expected = None
        if scenario:
            expected = next(
                (t for t in scenario.conversation_turns if t.turn_index == turn.turn_index), None
            )

        turn_failures = [r for r in results_by_turn.get(turn.turn_index, []) if not r.passed]
        marker = " ⚠️" if turn_failures else ""

        with st.expander(
            f"Turn {turn.turn_index}{marker} — {turn.user_message[:90]}",
            expanded=bool(turn_failures),
        ):
            st.markdown(f"**User:** {turn.user_message}")
            if expected and expected.expected_behavior:
                st.caption(f"Expected behaviour: {expected.expected_behavior}")
            if expected and expected.injected_failure.value != "none":
                target = (
                    expected.injected_failure_tool.value if expected.injected_failure_tool else "?"
                )
                st.warning(
                    f"Harness injected `{expected.injected_failure.value}` into "
                    f"`{target}` on this turn."
                )
            if expected and expected.contains_injection:
                st.error(
                    "Retrieved content on this turn carries an adversarial instruction: "
                    f"“{expected.injection_payload}”. The agent must treat it as data."
                )

            for call in turn.tool_calls:
                status = "✅" if call.succeeded else "❌"
                flag = "" if call.authorized else " 🚨 UNAUTHORISED"
                st.markdown(f"**{status} `{call.tool_name.value}`**{flag}")
                inner = st.columns(2)
                with inner[0]:
                    st.caption("Arguments")
                    st.json(call.arguments, expanded=False)
                with inner[1]:
                    st.caption("Result" if call.succeeded else "Error")
                    if call.succeeded:
                        st.json(call.result, expanded=False)
                    else:
                        st.error(f"{call.error_type}: {call.error}")
                meta = [f"latency {call.latency_ms:.0f} ms"]
                if call.retry_count:
                    meta.append(f"retries {call.retry_count}")
                if call.injected_failure.value != "none":
                    meta.append(f"injected {call.injected_failure.value}")
                st.caption(" · ".join(meta))

            st.markdown(f"**Assistant:** {turn.assistant_message or '_(no reply)_'}")

            if turn.clarification_requested:
                st.info("Agent asked the user to resolve a conflict.")
            if turn.refused_injection:
                st.success("Agent explicitly declined an instruction found in retrieved content.")

            if turn.workspace_after:
                st.caption("Agent state after this turn")
                st.json(
                    {
                        "facts": turn.workspace_after.facts,
                        "active_constraints": turn.workspace_after.active_constraints,
                        "pending_approval": turn.workspace_after.pending_approval,
                    },
                    expanded=False,
                )

            if turn_failures:
                st.markdown("**Evaluator findings for this turn**")
                for item in turn_failures:
                    icon = SEVERITY_ICON.get(item.severity.value, "⚪")
                    st.markdown(
                        f"{icon} **{item.evaluator_name}** "
                        f"(`{item.failure_category.value}`, {item.severity.value}) — "
                        f"{item.reasoning_summary}"
                    )
                    for evidence in item.evidence:
                        st.code(evidence.excerpt or "(empty)", language=None)
                        if evidence.detail:
                            st.caption(evidence.detail)

    st.divider()
    st.subheader("Session-level verdicts")

    kinds = {
        "deterministic": "Deterministic checks",
        "semantic": "Semantic diagnostics",
        "judge": "LLM judge",
    }
    tabs = st.tabs(list(kinds.values()))
    for tab, kind in zip(tabs, kinds, strict=True):
        with tab:
            subset = [item for item in session_level if item.evaluator_kind == kind]
            if not subset:
                st.info(f"No {kind} results for this session.")
                continue
            if kind == "judge":
                st.caption(
                    "Judge scores are reported alongside deterministic results and are "
                    "never blended into them, nor used to gate a release (ADR-002)."
                )
            if kind == "semantic":
                st.caption(
                    "Semantic scores use a lexical backend by default. They report a "
                    "number but never assert a failure."
                )
            for item in sorted(subset, key=lambda r: (r.passed, r.dimension.value)):
                icon = "✅" if item.passed else SEVERITY_ICON.get(item.severity.value, "⚪")
                st.markdown(
                    f"{icon} **{item.evaluator_name}** · {item.dimension.value} · "
                    f"score {item.score:.3f} · confidence {item.confidence:.2f}"
                )
                if item.reasoning_summary:
                    st.caption(item.reasoning_summary)
                if item.metadata.get("judge_model"):
                    st.caption(
                        f"judge {item.metadata['judge_model']} "
                        f"prompt {item.metadata.get('judge_prompt_version')} · "
                        f"{item.metadata.get('samples')} samples · "
                        f"spread {item.metadata.get('score_spread')}"
                    )

    st.divider()
    st.subheader("Final decision")
    st.write(
        {
            "overall_score": round(session.overall_score, 4),
            "passed": session.passed,
            "critical_failures": session.critical_failures,
            "failure_categories": session.failure_categories,
            "dimension_scores": {
                item.dimension.value: round(item.score, 4) for item in session.dimension_scores
            },
        }
    )


main()
