"""Blind human annotation interface.

The blindness is the point. Automated scores are not rendered anywhere on this page
until an annotation has been submitted, and each stored annotation records whether it
was collected blind. Non-blind annotations are excluded from agreement statistics
entirely — an annotator who has already seen the automated verdict is not an
independent rater, and pooling them would inflate every agreement number.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_access as data

from evalforge.ids import stable_id
from evalforge.schemas.annotation import DimensionRating, HumanAnnotation
from evalforge.schemas.common import Dimension, FailureCategory, Severity

st.set_page_config(page_title="Annotation", page_icon="✍️", layout="wide")

RATED_DIMENSIONS = (
    Dimension.TASK_COMPLETION,
    Dimension.CONTEXT_RETENTION,
    Dimension.INSTRUCTION_ADHERENCE,
    Dimension.RECOVERY_QUALITY,
)

RUBRIC_HELP = {
    Dimension.TASK_COMPLETION: (
        "1 = no progress on the objective, 5 = fully satisfied including later revisions."
    ),
    Dimension.CONTEXT_RETENTION: (
        "1 = critical earlier facts lost or invented, 5 = every fact carried forward correctly."
    ),
    Dimension.INSTRUCTION_ADHERENCE: (
        "1 = a persistent constraint ignored outright, 5 = every active constraint honoured."
    ),
    Dimension.RECOVERY_QUALITY: (
        "1 = invented results or abandoned the task, 5 = retried, adapted or asked."
    ),
}


def main() -> None:
    """Render the annotation interface."""
    st.title("Blind session annotation")
    st.caption("Read the session, score it, submit. Automated scores stay hidden until you do.")

    options = data.run_options()
    if not options:
        st.warning("No runs stored. Run `evalforge demo` first.")
        st.stop()

    with st.sidebar:
        label = st.selectbox("Run", list(options), key="annot_run")
        annotator = st.text_input(
            "Your annotator id",
            value="annotator_a",
            help="A pseudonym. At least two annotators are needed for a human-human ceiling.",
        )
        st.caption(
            "Agreement statistics need each session labelled by two independent "
            "annotators. Pick a distinct id per person."
        )

    run_id = options[label]
    if not annotator.strip():
        st.warning("Enter an annotator id to begin.")
        st.stop()

    store = data.get_store()
    done = store.annotated_session_ids(run_id, annotator.strip())
    sessions = data.sessions_of(run_id)
    pending = [item for item in sessions if item.session_id not in done]

    st.progress(
        len(done) / len(sessions) if sessions else 0.0,
        text=f"{len(done)}/{len(sessions)} sessions annotated by {annotator}",
    )

    if not pending:
        st.success("You have annotated every session in this run.")
        st.stop()

    session = pending[0]
    scenario = data.load_scenario(run_id, session.scenario_id)
    try:
        trace = data.load_trace(run_id, session.session_id)
    except Exception as exc:
        st.error(f"Could not load the trace for this session: {exc}")
        st.stop()

    st.subheader(scenario.name if scenario else session.scenario_id)
    if scenario:
        st.caption(
            f"{scenario.category.value.replace('_', ' ')} · {scenario.difficulty.value} · "
            f"{session.turn_count} turns"
        )
        st.markdown(f"**What this scenario tests:** {scenario.description}")

        with st.expander("Instructions the user gave (active constraints)", expanded=True):
            if scenario.active_constraints:
                for constraint in scenario.active_constraints:
                    st.markdown(
                        f"- **{constraint.description}** _(from turn {constraint.turn_added})_"
                    )
            else:
                st.write("No persistent constraints in this scenario.")

        with st.expander("Facts the agent should hold at the end", expanded=True):
            for key, fact in sorted(scenario.final_fact_values().items()):
                marker = " ⚠️ critical" if fact.critical else ""
                st.markdown(f"- `{key}` = **{fact.value}**{marker}")

    st.divider()
    st.subheader("Conversation")

    for turn in trace.turns:
        with st.chat_message("user"):
            st.markdown(f"**Turn {turn.turn_index}**")
            st.write(turn.user_message)
        with st.chat_message("assistant"):
            if turn.tool_calls:
                for call in turn.tool_calls:
                    status = "✅" if call.succeeded else "❌"
                    flag = "" if call.authorized else " 🚨 **unauthorised**"
                    with st.expander(
                        f"{status} tool: {call.tool_name.value}{flag}", expanded=False
                    ):
                        st.write("**Arguments**")
                        st.json(call.arguments, expanded=False)
                        if call.succeeded:
                            st.write("**Result**")
                            st.json(call.result, expanded=False)
                        else:
                            st.error(f"{call.error_type}: {call.error}")
            st.write(turn.assistant_message or "_(no reply)_")

        if scenario:
            expected = next(
                (t for t in scenario.conversation_turns if t.turn_index == turn.turn_index), None
            )
            if expected and expected.expected_behavior:
                st.caption(f"↳ Expected: {expected.expected_behavior}")

    st.divider()
    st.subheader("Your assessment")
    st.caption(
        "Judge the session as a whole, not the last message. Do not reward length — a "
        "short correct answer outranks a long hedged one."
    )

    started = st.session_state.setdefault(f"start_{session.session_id}", time.time())

    with st.form(key=f"annotation_{session.session_id}"):
        ratings: dict[Dimension, int] = {}
        columns = st.columns(len(RATED_DIMENSIONS))
        for column, dimension in zip(columns, RATED_DIMENSIONS, strict=True):
            with column:
                ratings[dimension] = st.slider(
                    dimension.value.replace("_", " ").title(),
                    min_value=1,
                    max_value=5,
                    value=3,
                    help=RUBRIC_HELP[dimension],
                )

        overall_pass = st.radio(
            "Would you accept this session as delivered?",
            options=[True, False],
            format_func=lambda value: "Yes — acceptable" if value else "No — not acceptable",
            horizontal=True,
        )

        categories = st.multiselect(
            "Failure categories you observed",
            options=[c.value for c in FailureCategory if c is not FailureCategory.NONE],
            help="Leave empty if the session was clean.",
        )

        severity = st.select_slider(
            "Worst severity observed",
            options=[s.value for s in Severity],
            value=Severity.INFO.value,
        )

        comments = st.text_area(
            "Comments",
            placeholder="What drove your verdict? Anything the automated checks might miss?",
        )

        submitted = st.form_submit_button("Submit annotation", type="primary")

    if not submitted:
        st.info(
            "🔒 Automated scores for this session are hidden until you submit. This "
            "annotation will be recorded as **blind** and is eligible for agreement "
            "statistics."
        )
        return

    annotation = HumanAnnotation(
        annotation_id=stable_id("annotation", run_id, session.session_id, annotator),
        run_id=run_id,
        session_id=session.session_id,
        scenario_id=session.scenario_id,
        annotator_id=annotator.strip(),
        ratings=[
            DimensionRating(dimension=dimension, rating=value)
            for dimension, value in ratings.items()
        ],
        overall_pass=bool(overall_pass),
        failure_categories=[FailureCategory(value) for value in categories],
        severity=Severity(severity),
        comments=comments,
        blind=True,
        duration_seconds=max(0.0, time.time() - started),
    )
    store.save_annotation(annotation)
    data.clear_caches()

    st.success("Annotation saved.")

    st.divider()
    st.subheader("Automated verdict for comparison")
    st.caption("Revealed only now that your judgement is recorded.")

    columns = st.columns(4)
    columns[0].metric("Automated score", f"{session.overall_score:.3f}")
    columns[1].metric("Automated verdict", "pass" if session.passed else "fail")
    columns[2].metric("Your verdict", "pass" if overall_pass else "fail")
    columns[3].metric("Critical failures", len(session.critical_failures))

    if session.passed != bool(overall_pass):
        st.warning(
            "You and the automated evaluators disagree on this session. That is a useful "
            "data point, not a mistake — it will appear on the alignment page for "
            "adjudication against the trace."
        )

    if session.failure_categories:
        st.write(f"**Automated failure categories:** {', '.join(session.failure_categories)}")
    if categories:
        st.write(f"**Your failure categories:** {', '.join(categories)}")

    if st.button("Annotate the next session"):
        st.rerun()


main()
