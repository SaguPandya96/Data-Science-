"""The tools Keel ships with: memory, calendar, tasks, goals, notes and email."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from keel.memory.embeddings import DEFAULT_ENCODER
from keel.memory.rerank import without_constraints
from keel.memory.retrieval import KeelMemory, bm25_scores
from keel.memory.store import KINDS, Memory
from keel.memory.text import canonical_key, expand, similar_keys
from keel.tools.registry import Tool, ToolContext, ToolError, parse_when, require_text

# Things that must never be written to long-term memory, whatever the model decides.
_SECRET_PATTERNS = (
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # payment card numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US social security numbers
    re.compile(r"\b(password|passcode|pin code|api key|secret key)\b", re.IGNORECASE),
)


# --- memory -----------------------------------------------------------------------------


def _remember(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    text = require_text(args["text"], "text", limit=500)
    if any(p.search(text) for p in _SECRET_PATTERNS):
        raise ToolError(
            "Refused: this looks like a password, card number or ID number. Keel never "
            "stores secrets. Tell the user it was not saved."
        )
    kind = args.get("kind", "fact")
    if kind not in KINDS:
        raise ToolError(f"kind must be one of {', '.join(KINDS)}")
    importance = args.get("importance")
    if args.get("key") and not args.get("new_key"):
        _check_key(ctx, args["key"])
    keyed = [m for m in ctx.memory.all() if m.key]
    try:
        memory, superseded = ctx.memory.add(
            text,
            kind=kind,
            key=args.get("key"),
            importance=importance,
            session_id=ctx.session_id,
        )
    except ValueError as error:
        raise ToolError(str(error)) from None
    result: dict[str, Any] = {"saved": memory.id, "key": canonical_key(memory.key)}
    if superseded:
        result["replaced"] = [{"id": m.id, "text": m.text} for m in superseded]
    elif memory.key is None and kind != "episode":
        result["hint"] = "No key given: if this detail can change later, save it with a key."
    elif memory.key is not None and canonical_key(memory.key) not in {
        canonical_key(m.key) for m in keyed
    }:
        match = _same_meaning(ctx, memory, keyed)
        if match is not None:
            result["possible_same_detail"] = {"key": match.key, "text": match.text}
            result["hint"] = (
                f"New key {memory.key!r} may describe the same detail as {match.key!r}. If "
                f"it does, forget memory {memory.id} and save it again with key "
                f"{match.key!r} so the old value is retired."
            )
    return result


def _check_key(ctx: ToolContext, key: str) -> None:
    """Refuse a new key that looks like a variant of one already in use.

    Forgetting relies on a changed detail being saved under the same key; a near-duplicate
    such as "city_home" for "home_city" would leave the old value active beside the new.
    """
    active = {canonical_key(m.key): m for m in ctx.memory.all() if m.key}
    canon = canonical_key(key)
    if canon in active:
        return
    similar = similar_keys(key, [k for k in active if k is not None])
    if similar:
        match = active[similar[0]]
        raise ToolError(
            f"Not saved: {key!r} looks like the existing key {similar[0]!r} "
            f"({match.text!r}). If this is the same detail, save it again with key "
            f"{similar[0]!r} so the old value is retired; if it is a different detail, "
            "save it again with new_key set to true."
        )


# Cosine between "key words: text" for a memory under a new key and for each memory under a
# key in use, with the shipped encoder. On the benchmark's 16 synonym keys (alt_key, saved
# with an update sentence) against the 16 real keys (with a first statement), the right
# key always scored highest; 13 of 16 reached 0.82, and no wrong key did. The threshold
# was chosen on those same pairs, so expect fewer hits in use. Unrelated details can still
# pass (a bedtime against wake_time scored 0.87), which is why this is a hint, not a refusal.
SAME_MEANING = 0.82


def _same_meaning(ctx: ToolContext, memory: Memory, keyed: list[Memory]) -> Memory | None:
    """The memory under a key in use that most likely describes the same detail, if any."""
    retriever = ctx.retriever
    base = getattr(retriever, "base", retriever)
    embedder = getattr(base, "embedder", None)
    # The threshold holds for the shipped encoder only.
    if not keyed or embedder is None or getattr(embedder, "name", None) != DEFAULT_ENCODER["name"]:
        return None
    prefix = getattr(embedder, "query_prefix", "")

    def line(m: Memory) -> str:
        return f"{prefix}{(canonical_key(m.key) or '').replace('_', ' ')}: {m.text}"

    vectors = np.asarray(embedder.embed([line(memory)] + [line(m) for m in keyed]))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = vectors[1:] @ vectors[0]
    best = int(np.argmax(scores))
    return keyed[best] if scores[best] >= SAME_MEANING else None


def _recall(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    query = require_text(args["query"], "query")
    limit = int(args.get("limit", 8))
    if not 1 <= limit <= 25:
        raise ToolError("limit must be between 1 and 25")
    # Same retriever as the context block, minus the always-on constraints.
    retriever = without_constraints(ctx.retriever or KeelMemory())
    found = retriever.retrieve(ctx.memory.all(), query, ctx.clock.now(), limit)
    return [
        {
            "id": m.id,
            "kind": m.kind,
            "key": m.key,
            "text": m.text,
            "date": f"{m.created_at:%Y-%m-%d}",
        }
        for m in found
    ]


def _forget(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    try:
        memory = ctx.memory.forget(int(args["memory_id"]))
    except KeyError as error:
        raise ToolError(str(error)) from None
    return {"forgotten": memory.id, "was": memory.text}


# --- calendar ---------------------------------------------------------------------------


def _events_between(ctx: ToolContext, start: datetime, end: datetime) -> list[dict[str, Any]]:
    rows = ctx.conn.execute(
        "SELECT * FROM events WHERE start < ? AND end > ? ORDER BY start",
        (end.isoformat(), start.isoformat()),
    ).fetchall()
    return [dict(r) for r in rows]


def _calendar_list(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    start = parse_when(args["start_date"], "start_date")
    end = parse_when(args.get("end_date", args["start_date"]), "end_date")
    if end <= start:
        end = start + timedelta(days=1)
    return _events_between(ctx, start, end)


def _calendar_add(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    title = require_text(args["title"], "title", limit=200)
    start = parse_when(args["start"], "start")
    end = parse_when(args["end"], "end")
    if end <= start:
        raise ToolError("end must be after start")
    conflicts = _events_between(ctx, start, end)
    if conflicts and not args.get("allow_conflict", False):
        listed = "; ".join(f"#{c['id']} {c['title']} {c['start']}-{c['end']}" for c in conflicts)
        raise ToolError(
            f"Conflicts with {listed}. Ask the user, then retry with allow_conflict=true "
            "or pick a free slot (calendar_find_free)."
        )
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO events (title, start, end, location, notes) VALUES (?, ?, ?, ?, ?)",
            (title, start.isoformat(), end.isoformat(), args.get("location"), args.get("notes")),
        )
    return {"event_id": cursor.lastrowid, "overlaps": [c["id"] for c in conflicts]}


def _calendar_find_free(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, str]]:
    day = parse_when(args["date"], "date").replace(hour=0, minute=0)
    minutes = int(args["duration_minutes"])
    if not 5 <= minutes <= 12 * 60:
        raise ToolError("duration_minutes must be between 5 and 720")
    earliest = int(args.get("earliest_hour", 8))
    latest = int(args.get("latest_hour", 18))
    if not 0 <= earliest < latest <= 24:
        raise ToolError("need 0 <= earliest_hour < latest_hour <= 24")
    window_start = day + timedelta(hours=earliest)
    window_end = day + timedelta(hours=latest)
    busy = [
        (datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"]))
        for e in _events_between(ctx, window_start, window_end)
    ]
    slots = []
    cursor = (
        max(window_start, ctx.clock.now()) if day.date() == ctx.clock.now().date() else window_start
    )
    for busy_start, busy_end in [*busy, (window_end, window_end)]:
        if busy_start - cursor >= timedelta(minutes=minutes):
            slots.append({"start": cursor.isoformat(), "end": busy_start.isoformat()})
        cursor = max(cursor, busy_end)
    return slots


# --- tasks and goals --------------------------------------------------------------------


def _task_add(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    title = require_text(args["title"], "title", limit=200)
    due = parse_when(args["due"], "due").isoformat() if args.get("due") else None
    priority = int(args.get("priority", 2))
    if priority not in (1, 2, 3):
        raise ToolError("priority is 1 (high), 2 (normal) or 3 (low)")
    goal_id = args.get("goal_id")
    if (
        goal_id is not None
        and not ctx.conn.execute("SELECT 1 FROM goals WHERE id = ?", (goal_id,)).fetchone()
    ):
        raise ToolError(f"no goal with id {goal_id}")
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO tasks (title, due, priority, goal_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (title, due, priority, goal_id, ctx.clock.now().isoformat()),
        )
    return {"task_id": cursor.lastrowid}


def _task_list(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    status = args.get("status", "open")
    if status not in ("open", "done", "all"):
        raise ToolError("status is open, done or all")
    where = "" if status == "all" else " WHERE status = ?"
    params: tuple[str, ...] = () if status == "all" else (status,)
    rows = ctx.conn.execute(
        f"SELECT * FROM tasks{where} ORDER BY due IS NULL, due, priority", params
    ).fetchall()
    return [dict(r) for r in rows]


def _task_complete(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    task_id = int(args["task_id"])
    with ctx.conn:
        updated = ctx.conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ? AND status = 'open'",
            (ctx.clock.now().isoformat(), task_id),
        ).rowcount
    if not updated:
        raise ToolError(f"no open task with id {task_id}")
    return {"completed": task_id}


def _goal_create(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    title = require_text(args["title"], "title", limit=200)
    target = (
        parse_when(args["target_date"], "target_date").date().isoformat()
        if args.get("target_date")
        else None
    )
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO goals (title, why, target_date, created_at) VALUES (?, ?, ?, ?)",
            (title, args.get("why"), target, ctx.clock.now().isoformat()),
        )
    return {"goal_id": cursor.lastrowid}


def _goal_plan(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    goal_id = int(args["goal_id"])
    goal = ctx.conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
    if goal is None:
        raise ToolError(f"no goal with id {goal_id}")
    milestones = args["milestones"]
    if not isinstance(milestones, list) or not milestones:
        raise ToolError("milestones must be a non-empty list of {title, due}")
    rows = []
    for item in milestones:
        if not isinstance(item, dict):
            raise ToolError("each milestone is an object with title and due")
        title = require_text(item.get("title"), "milestone title", limit=200)
        due = parse_when(item["due"], "milestone due").date() if item.get("due") else None
        if due and goal["target_date"] and due.isoformat() > goal["target_date"]:
            raise ToolError(f"milestone {title!r} is due after the goal's target date")
        rows.append((goal_id, title, due.isoformat() if due else None))
    with ctx.conn:
        ctx.conn.executemany("INSERT INTO milestones (goal_id, title, due) VALUES (?, ?, ?)", rows)
    return {"goal_id": goal_id, "milestones_added": len(rows)}


def _goal_checkin(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    goal_id = int(args["goal_id"])
    if not ctx.conn.execute("SELECT 1 FROM goals WHERE id = ?", (goal_id,)).fetchone():
        raise ToolError(f"no goal with id {goal_id}")
    progress = args.get("progress")
    if progress is not None and not 0 <= int(progress) <= 100:
        raise ToolError("progress is a percentage from 0 to 100")
    with ctx.conn:
        ctx.conn.execute(
            "INSERT INTO checkins (goal_id, note, progress, created_at) VALUES (?, ?, ?, ?)",
            (goal_id, require_text(args["note"], "note"), progress, ctx.clock.now().isoformat()),
        )
        if progress is not None and int(progress) >= 100:
            ctx.conn.execute("UPDATE goals SET status = 'done' WHERE id = ?", (goal_id,))
    return {"goal_id": goal_id, "logged": True}


def goal_overview(ctx: ToolContext) -> list[dict[str, Any]]:
    goals = []
    for goal in ctx.conn.execute("SELECT * FROM goals WHERE status = 'active' ORDER BY id"):
        item = dict(goal)
        item["milestones"] = [
            dict(m)
            for m in ctx.conn.execute(
                "SELECT id, title, due, done FROM milestones WHERE goal_id = ? ORDER BY due",
                (goal["id"],),
            )
        ]
        last = ctx.conn.execute(
            "SELECT note, progress, created_at FROM checkins WHERE goal_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (goal["id"],),
        ).fetchone()
        item["last_checkin"] = dict(last) if last else None
        goals.append(item)
    return goals


def _goal_list(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    return goal_overview(ctx)


# --- notes ------------------------------------------------------------------------------


def _note_write(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO notes (title, body, created_at) VALUES (?, ?, ?)",
            (
                require_text(args["title"], "title", limit=200),
                require_text(args["body"], "body", limit=20000),
                ctx.clock.now().isoformat(),
            ),
        )
    return {"note_id": cursor.lastrowid}


def _note_search(ctx: ToolContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    query = require_text(args["query"], "query")
    notes = [dict(r) for r in ctx.conn.execute("SELECT * FROM notes ORDER BY id")]
    scores = bm25_scores([f"{n['title']} {n['body']}" for n in notes], expand(query))
    ranked = sorted(zip(scores, notes, strict=True), key=lambda pair: pair[0], reverse=True)
    return [
        {"id": n["id"], "title": n["title"], "body": n["body"][:1500], "date": n["created_at"][:10]}
        for score, n in ranked[:5]
        if score > 0
    ]


# --- email ------------------------------------------------------------------------------

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _email_draft(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    to = require_text(args["to"], "to", limit=200)
    if not _EMAIL.match(to):
        raise ToolError(f"{to!r} is not an email address; ask the user for it")
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO drafts (to_addr, subject, body, created_at) VALUES (?, ?, ?, ?)",
            (
                to,
                require_text(args["subject"], "subject", limit=300),
                require_text(args["body"], "body", limit=20000),
                ctx.clock.now().isoformat(),
            ),
        )
    return {"draft_id": cursor.lastrowid, "status": "draft"}


def _email_send(ctx: ToolContext, args: dict[str, Any]) -> None:
    # Never called by the agent: outward tools become approvals. keel.approvals executes
    # this action only after the user approves it.
    raise AssertionError("email_send runs through keel.approvals")


def _daily_brief(ctx: ToolContext, args: dict[str, Any]) -> str:
    from keel.briefing import build_brief

    return build_brief(ctx).render()


def builtin_tools() -> list[Tool]:
    iso = {"type": "string", "description": "ISO date or date-time, e.g. 2026-10-02T14:30"}
    return [
        Tool(
            "remember",
            "Save a durable fact about the user to long-term memory. Give a short snake_case "
            "key for any detail that can change (home_city, employer, diet); reuse the "
            "existing key from the context when the detail changes, so the old value is "
            "retired. Never store passwords, card or ID numbers.",
            {
                "text": {"type": "string", "description": "One self-contained sentence."},
                "kind": {"type": "string", "enum": list(KINDS)},
                "key": {"type": "string"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                "new_key": {
                    "type": "boolean",
                    "description": "Set only when told a key looks like an existing one "
                    "but the detail really is different.",
                },
            },
            ("text", "kind"),
            "write",
            _remember,
        ),
        Tool(
            "recall",
            "Search long-term memory. Use it when the memories already in context don't "
            "answer the question; rephrase the topic in plain words (e.g. 'partner name' "
            "rather than 'anniversary dinner').",
            {"query": {"type": "string"}, "limit": {"type": "integer"}},
            ("query",),
            "read",
            _recall,
        ),
        Tool(
            "forget",
            "Delete a memory by id when the user asks you to forget something.",
            {"memory_id": {"type": "integer"}},
            ("memory_id",),
            "write",
            _forget,
        ),
        Tool(
            "calendar_list",
            "List calendar events that overlap a date range.",
            {"start_date": iso, "end_date": iso},
            ("start_date",),
            "read",
            _calendar_list,
        ),
        Tool(
            "calendar_add",
            "Add an event. Fails on a conflict unless allow_conflict is true.",
            {
                "title": {"type": "string"},
                "start": iso,
                "end": iso,
                "location": {"type": "string"},
                "notes": {"type": "string"},
                "allow_conflict": {"type": "boolean"},
            },
            ("title", "start", "end"),
            "write",
            _calendar_add,
        ),
        Tool(
            "calendar_find_free",
            "Find free slots of at least duration_minutes on a date.",
            {
                "date": iso,
                "duration_minutes": {"type": "integer"},
                "earliest_hour": {"type": "integer"},
                "latest_hour": {"type": "integer"},
            },
            ("date", "duration_minutes"),
            "read",
            _calendar_find_free,
        ),
        Tool(
            "task_add",
            "Add a to-do, optionally linked to a goal.",
            {
                "title": {"type": "string"},
                "due": iso,
                "priority": {"type": "integer", "enum": [1, 2, 3]},
                "goal_id": {"type": "integer"},
            },
            ("title",),
            "write",
            _task_add,
        ),
        Tool(
            "task_list",
            "List tasks by status (open, done, all).",
            {"status": {"type": "string", "enum": ["open", "done", "all"]}},
            (),
            "read",
            _task_list,
        ),
        Tool(
            "task_complete",
            "Mark a task done.",
            {"task_id": {"type": "integer"}},
            ("task_id",),
            "write",
            _task_complete,
        ),
        Tool(
            "goal_create",
            "Create a long-term goal. Follow up with goal_plan to break it into milestones.",
            {"title": {"type": "string"}, "why": {"type": "string"}, "target_date": iso},
            ("title",),
            "write",
            _goal_create,
        ),
        Tool(
            "goal_plan",
            "Add dated milestones to a goal. None may fall after the goal's target date.",
            {
                "goal_id": {"type": "integer"},
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}, "due": iso},
                        "required": ["title"],
                    },
                },
            },
            ("goal_id", "milestones"),
            "write",
            _goal_plan,
        ),
        Tool(
            "goal_checkin",
            "Log progress on a goal (progress is a percentage; 100 completes it).",
            {
                "goal_id": {"type": "integer"},
                "note": {"type": "string"},
                "progress": {"type": "integer"},
            },
            ("goal_id", "note"),
            "write",
            _goal_checkin,
        ),
        Tool(
            "goal_list",
            "Active goals with milestones and the latest check-in.",
            {},
            (),
            "read",
            _goal_list,
        ),
        Tool(
            "note_write",
            "Save a note (meeting notes, lists, anything longer than a memory).",
            {"title": {"type": "string"}, "body": {"type": "string"}},
            ("title", "body"),
            "write",
            _note_write,
        ),
        Tool(
            "note_search",
            "Search saved notes.",
            {"query": {"type": "string"}},
            ("query",),
            "read",
            _note_search,
        ),
        Tool(
            "email_draft",
            "Write an email draft. Drafts are only saved, never sent.",
            {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            ("to", "subject", "body"),
            "write",
            _email_draft,
        ),
        Tool(
            "email_send",
            "Ask to send a saved draft. This queues it for the user's approval; it is not "
            "sent until they approve.",
            {"draft_id": {"type": "integer"}},
            ("draft_id",),
            "outward",
            _email_send,
        ),
        Tool(
            "daily_brief",
            "Today's agenda, due and overdue tasks, goals needing a check-in, and pending "
            "approvals.",
            {},
            (),
            "read",
            _daily_brief,
        ),
    ]
