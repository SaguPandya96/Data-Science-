"""The approval queue: the only way an outward action ever happens.

The agent can ask; only the user, through the CLI or the app, can approve. Approved emails
are written to an outbox folder as .eml files rather than sent over SMTP: wiring a real mail
account is left to the user, and the gate in front of it is what this module is about.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from keel.clock import Clock
from keel.tools.registry import ToolContext, ToolError


@dataclass(frozen=True)
class Approval:
    id: int
    action: str
    summary: str
    payload: dict[str, Any]
    status: str
    created_at: str
    result: str | None


def request_approval(ctx: ToolContext, action: str, payload: dict[str, Any]) -> int:
    summary = _describe(ctx.conn, action, payload)
    with ctx.conn:
        cursor = ctx.conn.execute(
            "INSERT INTO approvals (action, summary, payload, created_at) VALUES (?, ?, ?, ?)",
            (action, summary, json.dumps(payload, sort_keys=True), ctx.clock.now().isoformat()),
        )
    return int(cursor.lastrowid or 0)


def _describe(conn: sqlite3.Connection, action: str, payload: dict[str, Any]) -> str:
    if action == "email_send":
        draft = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (payload.get("draft_id"),)
        ).fetchone()
        if draft is None:
            raise ToolError(f"no draft with id {payload.get('draft_id')}; write one first")
        if draft["status"] != "draft":
            raise ToolError(f"draft {draft['id']} is already {draft['status']}")
        waiting = _pending_for_draft(conn, draft["id"])
        if waiting is not None:
            raise ToolError(
                f"draft {draft['id']} is already waiting for approval #{waiting}; "
                "tell the user it is waiting"
            )
        return f"Send email to {draft['to_addr']}: {draft['subject']!r}"
    raise ToolError(f"no approval flow for {action}")


def _pending_for_draft(conn: sqlite3.Connection, draft_id: int) -> int | None:
    """The id of a pending send for this draft, if one is already queued."""
    for row in conn.execute(
        "SELECT id, payload FROM approvals WHERE status = 'pending' AND action = 'email_send'"
    ):
        if json.loads(row["payload"]).get("draft_id") == draft_id:
            return int(row["id"])
    return None


def _row(row: sqlite3.Row) -> Approval:
    return Approval(
        id=row["id"],
        action=row["action"],
        summary=row["summary"],
        payload=json.loads(row["payload"]),
        status=row["status"],
        created_at=row["created_at"],
        result=row["result"],
    )


def pending(conn: sqlite3.Connection) -> list[Approval]:
    rows = conn.execute("SELECT * FROM approvals WHERE status = 'pending' ORDER BY id")
    return [_row(r) for r in rows]


def get(conn: sqlite3.Connection, approval_id: int) -> Approval:
    row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if row is None:
        raise KeyError(f"no approval with id {approval_id}")
    return _row(row)


def approve(conn: sqlite3.Connection, approval_id: int, clock: Clock, outbox: Path) -> Approval:
    """Carry out a pending action the user has approved."""
    approval = get(conn, approval_id)
    if approval.status != "pending":
        raise ValueError(f"approval {approval_id} is already {approval.status}")
    if approval.action != "email_send":
        raise ValueError(f"cannot execute {approval.action}")
    draft = conn.execute(
        "SELECT * FROM drafts WHERE id = ?", (approval.payload["draft_id"],)
    ).fetchone()
    # Checked again here, not only when the send was queued: a database written by an
    # older version can hold two pending sends for one draft, and only one may go out.
    if draft is None:
        raise ValueError(f"approval {approval_id} points to a draft that no longer exists")
    if draft["status"] != "draft":
        raise ValueError(f"draft {draft['id']} is already {draft['status']}; not sending it again")
    message = EmailMessage()
    message["To"] = draft["to_addr"]
    message["Subject"] = draft["subject"]
    message.set_content(draft["body"])
    outbox = outbox.expanduser()
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"draft-{draft['id']}.eml"
    path.write_bytes(bytes(message))
    now = clock.now().isoformat()
    with conn:
        conn.execute("UPDATE drafts SET status = 'sent' WHERE id = ?", (draft["id"],))
        conn.execute(
            "UPDATE approvals SET status = 'executed', decided_at = ?, result = ? WHERE id = ?",
            (now, str(path), approval_id),
        )
    return get(conn, approval_id)


def reject(conn: sqlite3.Connection, approval_id: int, clock: Clock) -> Approval:
    approval = get(conn, approval_id)
    if approval.status != "pending":
        raise ValueError(f"approval {approval_id} is already {approval.status}")
    with conn:
        conn.execute(
            "UPDATE approvals SET status = 'rejected', decided_at = ? WHERE id = ?",
            (clock.now().isoformat(), approval_id),
        )
    return get(conn, approval_id)
