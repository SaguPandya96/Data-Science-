"""Long-term memory: what the agent knows about its user, and what it has since unlearned.

Every memory can carry a key naming the personal detail it describes ("home_city",
"employer"). Writing a new memory under a key that already has an active memory marks the
old one as superseded. Superseded memories are kept for the audit trail but are never
shown to the model by Keel's retriever, which is what stops a user's old city from
leaking into new plans.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from keel.clock import Clock, SystemClock
from keel.memory.text import canonical_key

KINDS = ("fact", "preference", "constraint", "goal", "episode")

DEFAULT_IMPORTANCE = {"constraint": 5, "goal": 4, "fact": 3, "preference": 3, "episode": 1}


@dataclass(frozen=True)
class Memory:
    id: int
    kind: str
    key: str | None
    text: str
    importance: int
    created_at: datetime
    superseded_by: int | None = None
    deleted: bool = False

    @property
    def active(self) -> bool:
        return self.superseded_by is None and not self.deleted

    def render(self) -> str:
        """One line for the prompt: date, kind and text."""
        return f"[{self.created_at:%Y-%m-%d}] ({self.kind}) {self.text}"


def _from_row(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        kind=row["kind"],
        key=row["key"],
        text=row["text"],
        importance=row["importance"],
        created_at=datetime.fromisoformat(row["created_at"]),
        superseded_by=row["superseded_by"],
        deleted=bool(row["deleted"]),
    )


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection, clock: Clock | None = None) -> None:
        self.conn = conn
        self.clock = clock or SystemClock()

    def add(
        self,
        text: str,
        *,
        kind: str = "fact",
        key: str | None = None,
        importance: int | None = None,
        session_id: str | None = None,
    ) -> tuple[Memory, list[Memory]]:
        """Store a memory. Returns it and any memories it superseded."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
        text = text.strip()
        if not text:
            raise ValueError("memory text is empty")
        importance = DEFAULT_IMPORTANCE[kind] if importance is None else importance
        if not 1 <= importance <= 5:
            raise ValueError("importance must be between 1 and 5")
        canon = canonical_key(key)
        now = self.clock.now().isoformat()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO memories (kind, key, canonical_key, text, importance, created_at,"
                " session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, key, canon, text, importance, now, session_id),
            )
            new_id = int(cursor.lastrowid or 0)
            superseded: list[Memory] = []
            if canon is not None:
                rows = self.conn.execute(
                    "SELECT * FROM memories WHERE canonical_key = ? AND id != ?"
                    " AND superseded_by IS NULL AND deleted = 0",
                    (canon, new_id),
                ).fetchall()
                superseded = [_from_row(r) for r in rows]
                self.conn.executemany(
                    "UPDATE memories SET superseded_by = ? WHERE id = ?",
                    [(new_id, m.id) for m in superseded],
                )
        return self.get(new_id), superseded

    def get(self, memory_id: int) -> Memory:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(f"no memory with id {memory_id}")
        return _from_row(row)

    def forget(self, memory_id: int) -> Memory:
        """Delete a memory at the user's request. It stops being retrievable at once."""
        memory = self.get(memory_id)
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET deleted = 1, text = '[forgotten]' WHERE id = ?",
                (memory_id,),
            )
        return memory

    def all(self, *, include_inactive: bool = False) -> list[Memory]:
        """Memories in the order they were written."""
        where = "" if include_inactive else " WHERE superseded_by IS NULL AND deleted = 0"
        rows = self.conn.execute(f"SELECT * FROM memories{where} ORDER BY id").fetchall()
        return [_from_row(r) for r in rows]

    def history(self, memory_id: int) -> list[Memory]:
        """The chain of earlier values a memory replaced, oldest first."""
        chain: list[Memory] = []
        current = [memory_id]
        while current:
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE superseded_by IN ({','.join('?' * len(current))})",
                current,
            ).fetchall()
            found = [_from_row(r) for r in rows]
            chain = found + chain
            current = [m.id for m in found]
        return chain
