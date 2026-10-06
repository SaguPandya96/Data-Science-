from __future__ import annotations

import json
from pathlib import Path

from keel import approvals


def run(toolbox, name, **args):
    return toolbox.run("t1", name, args)


def test_remember_supersedes_and_reports_it(toolbox):
    run(toolbox, "remember", text="Lives in Denver.", kind="fact", key="home_city")
    result = run(toolbox, "remember", text="Moved to Austin.", kind="fact", key="home_city")
    payload = json.loads(result.output)
    assert payload["replaced"][0]["text"] == "Lives in Denver."


def test_remember_refuses_secrets(toolbox, store):
    for text in ("My card is 4111 1111 1111 1111", "Email password is hunter2"):
        result = run(toolbox, "remember", text=text, kind="fact")
        assert result.is_error
        assert "never stores secrets" in result.output
    assert store.all() == []


def test_invalid_input_is_reported_not_raised(toolbox):
    result = run(toolbox, "calendar_add", title="x")
    assert result.is_error and "missing" in result.output
    result = run(toolbox, "task_add", title="x", colour="red")
    assert result.is_error and "unexpected colour" in result.output
    assert toolbox.run("t", "launch_rocket", {}).is_error


def test_calendar_conflicts_and_free_slots(toolbox):
    ok = run(
        toolbox, "calendar_add", title="Standup", start="2026-10-02T10:00", end="2026-10-02T10:30"
    )
    assert not ok.is_error
    clash = run(
        toolbox, "calendar_add", title="Dentist", start="2026-10-02T10:15", end="2026-10-02T11:00"
    )
    assert clash.is_error and "Conflicts with" in clash.output
    forced = run(
        toolbox,
        "calendar_add",
        title="Dentist",
        start="2026-10-02T10:15",
        end="2026-10-02T11:00",
        allow_conflict=True,
    )
    assert json.loads(forced.output)["overlaps"] == [1]
    slots = json.loads(
        run(
            toolbox,
            "calendar_find_free",
            date="2026-10-02",
            duration_minutes=60,
            earliest_hour=9,
            latest_hour=13,
        ).output
    )
    assert slots == [
        {"start": "2026-10-02T09:00:00", "end": "2026-10-02T10:00:00"},
        {"start": "2026-10-02T11:00:00", "end": "2026-10-02T13:00:00"},
    ]


def test_bad_dates_explain_the_format(toolbox):
    result = run(toolbox, "calendar_list", start_date="next tuesday")
    assert result.is_error and "2026-10-02T14:30" in result.output


def test_goal_plan_rejects_milestones_after_target(toolbox):
    run(toolbox, "goal_create", title="Learn Spanish", target_date="2026-12-31")
    bad = run(
        toolbox, "goal_plan", goal_id=1, milestones=[{"title": "B1 exam", "due": "2027-02-01"}]
    )
    assert bad.is_error and "after the goal's target date" in bad.output
    good = run(
        toolbox,
        "goal_plan",
        goal_id=1,
        milestones=[{"title": "100 words", "due": "2026-10-20"}, {"title": "A2"}],
    )
    assert json.loads(good.output)["milestones_added"] == 2
    run(toolbox, "goal_checkin", goal_id=1, note="Done", progress=100)
    assert json.loads(run(toolbox, "goal_list").output) == []


def test_tasks_lifecycle(toolbox):
    run(toolbox, "task_add", title="Call plumber", due="2026-10-01", priority=1)
    assert json.loads(run(toolbox, "task_list").output)[0]["title"] == "Call plumber"
    assert not run(toolbox, "task_complete", task_id=1).is_error
    assert run(toolbox, "task_complete", task_id=1).is_error
    assert json.loads(run(toolbox, "task_list", status="done").output)[0]["id"] == 1


def test_notes_search(toolbox):
    run(toolbox, "note_write", title="Trip ideas", body="Lisbon in spring, Kyoto in autumn")
    run(toolbox, "note_write", title="Groceries", body="eggs, oats")
    found = json.loads(run(toolbox, "note_search", query="Kyoto trip").output)
    assert [n["title"] for n in found] == ["Trip ideas"]


def test_email_send_only_queues_an_approval(toolbox, conn, clock, tmp_path: Path):
    assert run(toolbox, "email_draft", to="not-an-address", subject="s", body="b").is_error
    run(toolbox, "email_draft", to="sam@example.com", subject="Dinner", body="Friday?")
    result = run(toolbox, "email_send", draft_id=1)
    assert json.loads(result.output)["status"] == "pending_approval"
    assert result.approval_id == 1
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "draft"
    assert not list(tmp_path.iterdir())

    done = approvals.approve(conn, 1, clock, tmp_path)
    assert done.status == "executed"
    assert "Subject: Dinner" in (tmp_path / "draft-1.eml").read_text()
    # A sent draft can't be queued again.
    assert run(toolbox, "email_send", draft_id=1).is_error


def test_reject_and_missing_draft(toolbox, conn, clock):
    assert run(toolbox, "email_send", draft_id=99).is_error
    run(toolbox, "email_draft", to="a@b.co", subject="s", body="b")
    run(toolbox, "email_send", draft_id=1)
    assert approvals.reject(conn, 1, clock).status == "rejected"
    assert approvals.pending(conn) == []


def test_every_call_is_audited(toolbox, conn):
    run(toolbox, "task_list")
    run(toolbox, "task_complete", task_id=5)
    rows = conn.execute("SELECT tool, is_error FROM audit ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("task_list", 0), ("task_complete", 1)]


def test_tool_definitions_are_sorted_and_closed(toolbox):
    names = [d["name"] for d in toolbox.definitions()]
    assert names == sorted(names)
    assert all(d["input_schema"]["additionalProperties"] is False for d in toolbox.definitions())


def test_remember_refuses_a_near_duplicate_key_until_confirmed(toolbox, store):
    run(toolbox, "remember", text="Lives in Denver.", kind="fact", key="home_city")
    result = run(toolbox, "remember", text="Moved to Austin.", kind="fact", key="city_home")
    assert result.is_error
    assert "'home_city'" in result.output and "Lives in Denver." in result.output
    assert len(store.all()) == 1
    # Saved under the existing key, the new value retires the old one.
    payload = json.loads(
        run(toolbox, "remember", text="Moved to Austin.", kind="fact", key="home_city").output
    )
    assert payload["replaced"][0]["text"] == "Lives in Denver."
    # A different detail that only looks similar goes through once confirmed.
    result = run(
        toolbox, "remember", text="Grew up in Boise.", kind="fact", key="city_home", new_key=True
    )
    assert not result.is_error
    assert {m.key for m in store.all()} == {"home_city", "city_home"}


def test_remember_allows_unrelated_and_respelled_keys(toolbox, store):
    run(toolbox, "remember", text="Works at Acme.", kind="fact", key="employer")
    assert not run(toolbox, "remember", text="Vegan.", kind="preference", key="diet").is_error
    assert not run(toolbox, "remember", text="Works at Beta.", kind="fact", key="Employer").is_error
    assert len(store.all()) == 2
