"""SQLite storage. One file holds everything the agent knows about its user."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,
    key           TEXT,
    canonical_key TEXT,
    text          TEXT NOT NULL,
    importance    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    superseded_by INTEGER REFERENCES memories(id),
    deleted       INTEGER NOT NULL DEFAULT 0,
    session_id    TEXT
);
CREATE INDEX IF NOT EXISTS memories_active ON memories(canonical_key)
    WHERE superseded_by IS NULL AND deleted = 0;

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY,
    title    TEXT NOT NULL,
    start    TEXT NOT NULL,
    end      TEXT NOT NULL,
    location TEXT,
    notes    TEXT
);

CREATE TABLE IF NOT EXISTS goals (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    why         TEXT,
    target_date TEXT,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS milestones (
    id      INTEGER PRIMARY KEY,
    goal_id INTEGER NOT NULL REFERENCES goals(id),
    title   TEXT NOT NULL,
    due     TEXT,
    done    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkins (
    id         INTEGER PRIMARY KEY,
    goal_id    INTEGER NOT NULL REFERENCES goals(id),
    note       TEXT NOT NULL,
    progress   INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    due          TEXT,
    priority     INTEGER NOT NULL DEFAULT 2,
    status       TEXT NOT NULL DEFAULT 'open',
    goal_id      INTEGER REFERENCES goals(id),
    created_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drafts (
    id         INTEGER PRIMARY KEY,
    to_addr    TEXT NOT NULL,
    subject    TEXT NOT NULL,
    body       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id         INTEGER PRIMARY KEY,
    action     TEXT NOT NULL,
    summary    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    result     TEXT
);

CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    session_id TEXT,
    tool       TEXT NOT NULL,
    input      TEXT NOT NULL,
    output     TEXT NOT NULL,
    is_error   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    reflected  INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open (and if needed create) a Keel database."""
    if str(path) != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        path = Path(path).expanduser()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
