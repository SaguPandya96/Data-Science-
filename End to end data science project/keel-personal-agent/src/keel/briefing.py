"""The daily brief: what Keel brings up without being asked.

Computed from the database alone, so it can run from cron each morning with no model call
and no cost (``keel brief``). The chat agent can also read it through the ``daily_brief``
tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from keel import approvals
from keel.tools.builtin import goal_overview
from keel.tools.registry import ToolContext

STALE_GOAL_DAYS = 7
MILESTONE_HORIZON_DAYS = 7


@dataclass
class Brief:
    day: datetime
    events: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    overdue: list[dict[str, Any]] = field(default_factory=list)
    due_today: list[dict[str, Any]] = field(default_factory=list)
    milestones_soon: list[dict[str, Any]] = field(default_factory=list)
    stale_goals: list[dict[str, Any]] = field(default_factory=list)
    pending_approvals: list[approvals.Approval] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.events,
                self.overdue,
                self.due_today,
                self.milestones_soon,
                self.stale_goals,
                self.pending_approvals,
            )
        )

    def render(self) -> str:
        lines = [f"Brief for {self.day:%A %d %B %Y}"]
        if self.empty:
            return lines[0] + "\nNothing scheduled, nothing due. A clear day."
        if self.events:
            lines.append("\nToday:")
            lines += [
                f"  {e['start'][11:16]}-{e['end'][11:16]}  {e['title']}"
                + (f" ({e['location']})" if e.get("location") else "")
                for e in self.events
            ]
        if self.conflicts:
            lines.append("\nClashes:")
            lines += [f"  {a} overlaps {b}" for a, b in self.conflicts]
        if self.overdue:
            lines.append("\nOverdue:")
            lines += [f"  #{t['id']} {t['title']} (due {t['due'][:10]})" for t in self.overdue]
        if self.due_today:
            lines.append("\nDue today:")
            lines += [f"  #{t['id']} {t['title']}" for t in self.due_today]
        if self.milestones_soon:
            lines.append("\nMilestones this week:")
            lines += [f"  {m['goal']}: {m['title']} (by {m['due']})" for m in self.milestones_soon]
        if self.stale_goals:
            lines.append(f"\nNo check-in for {STALE_GOAL_DAYS}+ days:")
            lines += [f"  #{g['id']} {g['title']}" for g in self.stale_goals]
        if self.pending_approvals:
            lines.append("\nWaiting for your approval:")
            lines += [f"  #{a.id} {a.summary}" for a in self.pending_approvals]
        return "\n".join(lines)


def build_brief(ctx: ToolContext) -> Brief:
    now = ctx.clock.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    brief = Brief(day=start)

    brief.events = [
        dict(r)
        for r in ctx.conn.execute(
            "SELECT * FROM events WHERE start < ? AND end > ? ORDER BY start",
            (end.isoformat(), start.isoformat()),
        )
    ]
    for i, first in enumerate(brief.events):
        for second in brief.events[i + 1 :]:
            if second["start"] < first["end"]:
                brief.conflicts.append((first["title"], second["title"]))

    for task in ctx.conn.execute(
        "SELECT * FROM tasks WHERE status = 'open' AND due IS NOT NULL ORDER BY due, priority"
    ):
        if task["due"] < start.isoformat():
            brief.overdue.append(dict(task))
        elif task["due"] < end.isoformat():
            brief.due_today.append(dict(task))

    horizon = (start + timedelta(days=MILESTONE_HORIZON_DAYS)).date().isoformat()
    for goal in goal_overview(ctx):
        for milestone in goal["milestones"]:
            if not milestone["done"] and milestone["due"] and milestone["due"] <= horizon:
                brief.milestones_soon.append({**milestone, "goal": goal["title"]})
        last = goal["last_checkin"]["created_at"] if goal["last_checkin"] else goal["created_at"]
        if now - datetime.fromisoformat(last) >= timedelta(days=STALE_GOAL_DAYS):
            brief.stale_goals.append(goal)

    brief.pending_approvals = approvals.pending(ctx.conn)
    return brief
