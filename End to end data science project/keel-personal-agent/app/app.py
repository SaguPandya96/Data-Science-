"""Keel in the browser: chat, today's brief, memory, goals and the approval queue.

    KEEL_DB=~/.keel/keel.db streamlit run app/app.py

Chat needs an Anthropic API key. Everything else reads the local database and works
without one; try it on the demo user (`keel --db ~/.keel/demo.db demo`).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from keel import approvals
from keel.agent import Agent
from keel.briefing import build_brief
from keel.clock import SystemClock
from keel.db import connect
from keel.memory.store import MemoryStore
from keel.model import ClaudeModel, api_key_available
from keel.tools.builtin import goal_overview
from keel.tools.registry import ToolContext

DB = Path(os.environ.get("KEEL_DB", "~/.keel/keel.db")).expanduser()
OUTBOX = Path(os.environ.get("KEEL_OUTBOX", "~/.keel/outbox")).expanduser()

st.set_page_config(page_title="Keel", page_icon="⚓", layout="wide")


@st.cache_resource
def _conn():  # type: ignore[no-untyped-def]
    return connect(DB)


conn = _conn()
clock = SystemClock()
store = MemoryStore(conn, clock)
ctx = ToolContext(conn, clock, store)

st.title("Keel")
st.caption(f"Your personal agent · {DB}")

chat_tab, today_tab, memory_tab, goals_tab, approvals_tab, audit_tab = st.tabs(
    ["Chat", "Today", "Memory", "Goals & tasks", "Approvals", "Activity"]
)

with chat_tab:
    if not api_key_available():
        st.info("Set ANTHROPIC_API_KEY to chat. The other tabs work without it.")
    else:
        if "agent" not in st.session_state:
            st.session_state.agent = Agent(conn, ClaudeModel())
            st.session_state.log = []
        for role, content in st.session_state.log:
            st.chat_message(role).markdown(content)
        if prompt := st.chat_input("Ask Keel to plan, remember or schedule something"):
            st.chat_message("user").markdown(prompt)
            with st.spinner("Thinking"):
                turn = st.session_state.agent.send(prompt)
            used = ", ".join(r.name + (" ✗" if r.is_error else "") for r in turn.tool_results)
            answer = turn.text + (f"\n\n_Tools: {used}_" if used else "")
            if turn.approvals:
                answer += "\n\n**Waiting for your approval** in the Approvals tab."
            st.chat_message("assistant").markdown(answer)
            st.session_state.log += [("user", prompt), ("assistant", answer)]
        if st.session_state.log and st.button("End session and save what I learned"):
            st.session_state.agent.reflect()
            del st.session_state["agent"]
            st.rerun()

with today_tab:
    st.code(build_brief(ctx).render(), language=None)

with memory_tab:
    show_all = st.toggle("Show replaced and forgotten memories")
    memories = store.all(include_inactive=show_all)
    if not memories:
        st.write("Nothing remembered yet.")
    for memory in reversed(memories):
        left, right = st.columns([6, 1])
        status = "" if memory.active else (" · forgotten" if memory.deleted else " · replaced")
        key = f" · `{memory.key}`" if memory.key else ""
        left.markdown(
            f"**{memory.text}**  \n{memory.created_at:%d %b %Y} · {memory.kind}{key}{status}"
        )
        if memory.active and right.button("Forget", key=f"forget-{memory.id}"):
            store.forget(memory.id)
            st.rerun()

with goals_tab:
    for goal in goal_overview(ctx):
        st.subheader(goal["title"])
        if goal["why"]:
            st.caption(goal["why"])
        for milestone in goal["milestones"]:
            mark = "✅" if milestone["done"] else "⬜"
            st.write(f"{mark} {milestone['title']} — {milestone['due'] or 'no date'}")
        if goal["last_checkin"]:
            st.write(f"Last check-in: {goal['last_checkin']['note']}")
    st.subheader("Open tasks")
    for task in conn.execute(
        "SELECT * FROM tasks WHERE status = 'open' ORDER BY due IS NULL, due, priority"
    ):
        st.write(f"#{task['id']} {task['title']} · due {(task['due'] or '—')[:16]}")

with approvals_tab:
    items = approvals.pending(conn)
    if not items:
        st.write("Nothing waiting for approval.")
    for item in items:
        st.markdown(f"**#{item.id} {item.summary}**")
        if item.action == "email_send":
            draft = conn.execute(
                "SELECT * FROM drafts WHERE id = ?", (item.payload["draft_id"],)
            ).fetchone()
            st.text(draft["body"])
        yes, no, _ = st.columns([1, 1, 6])
        if yes.button("Approve", key=f"approve-{item.id}"):
            done = approvals.approve(conn, item.id, clock, OUTBOX)
            st.success(f"Written to {done.result}")
            st.rerun()
        if no.button("Reject", key=f"reject-{item.id}"):
            approvals.reject(conn, item.id, clock)
            st.rerun()

with audit_tab:
    rows = conn.execute(
        "SELECT ts, tool, input, is_error FROM audit ORDER BY id DESC LIMIT 100"
    ).fetchall()
    st.dataframe(
        [
            {
                "time": r["ts"][:16],
                "tool": r["tool"],
                "input": r["input"],
                "error": bool(r["is_error"]),
            }
            for r in rows
        ],
        width="stretch",
    )
