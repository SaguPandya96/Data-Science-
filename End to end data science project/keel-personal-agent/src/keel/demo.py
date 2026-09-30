"""A made-up user, so the brief, the app and the memory view have something to show."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from keel.clock import Clock, FixedClock
from keel.memory.store import MemoryStore
from keel.tools.builtin import builtin_tools
from keel.tools.registry import Toolbox, ToolContext


def seed(conn: sqlite3.Connection, clock: Clock) -> None:
    now = clock.now().replace(second=0, microsecond=0)
    today = now.replace(hour=0, minute=0)
    past = FixedClock(now - timedelta(days=60))
    memory = MemoryStore(conn, past)

    def at(days_ago: int) -> None:
        past.set(now - timedelta(days=days_ago))

    at(60)
    memory.add("Lives in Denver.", kind="fact", key="home_city")
    memory.add("Works as a data analyst at Helio Health.", kind="fact", key="job")
    memory.add("Severely allergic to peanuts.", kind="constraint", key="allergy")
    memory.add("Partner's name is Sam.", kind="fact", key="partner_name")
    at(45)
    memory.add("Prefers no meetings before 10 am.", kind="preference", key="meeting_hours")
    memory.add("Goes to the gym Tuesday and Friday evenings.", kind="preference", key="gym_days")
    memory.add("Asked for a sourdough recipe.", kind="episode")
    at(20)
    memory.add("Moved to Austin for the new job.", kind="fact", key="home_city")
    memory.add("Started as a senior data scientist at Brightline.", kind="fact", key="job")
    memory.add("Keeps eating out under $300 a month.", kind="constraint", key="dining_budget")
    at(6)
    memory.add("Wants to run a half marathon in March.", kind="goal", key="race_goal")

    ctx = ToolContext(conn, clock, MemoryStore(conn, clock), "demo")
    tools = Toolbox(ctx)
    for tool in builtin_tools():
        tools.register(tool)

    def call(name: str, **args: object) -> None:
        result = tools.run("demo", name, dict(args))
        if result.is_error:
            raise RuntimeError(f"demo seed failed on {name}: {result.output}")

    def day(offset: int, hour: int, minute: int = 0) -> str:
        return (today + timedelta(days=offset, hours=hour, minutes=minute)).isoformat(
            timespec="minutes"
        )

    call("calendar_add", title="Team standup", start=day(0, 10), end=day(0, 10, 30))
    call("calendar_add", title="1:1 with manager", start=day(0, 14), end=day(0, 14, 45))
    call("calendar_add", title="Dentist", start=day(1, 9), end=day(1, 10), location="Oak St")
    call(
        "goal_create",
        title="Run a half marathon",
        why="Get fit before summer",
        target_date=(today + timedelta(days=160)).date().isoformat(),
    )
    call(
        "goal_plan",
        goal_id=1,
        milestones=[
            {"title": "Run 5K without stopping", "due": day(5, 0)[:10]},
            {"title": "Run 10K", "due": day(50, 0)[:10]},
            {"title": "Run 16K long run", "due": day(110, 0)[:10]},
        ],
    )
    call("task_add", title="Buy running shoes", due=day(-1, 18), priority=1, goal_id=1)
    call("task_add", title="Book physio check", due=day(0, 17), priority=2)
    call("task_add", title="Plan Sam's birthday dinner (peanut-free!)", due=day(9, 18))
    call(
        "email_draft",
        to="coach@example.com",
        subject="Half marathon plan",
        body="Hi! Could we set up a first session next week? Tuesdays and Fridays are gym "
        "days, so Monday or Wednesday evening works best.",
    )
    call("email_send", draft_id=1)
    conn.execute("UPDATE goals SET created_at = ?", ((now - timedelta(days=10)).isoformat(),))
    conn.commit()


def is_empty(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def seed_if_empty(conn: sqlite3.Connection, clock: Clock) -> bool:
    if not is_empty(conn):
        return False
    seed(conn, clock)
    return True
