"""Apply the numbered SQL files in migrations/ to a database, in order, once each.

    python -m rebalancer.migrate              # apply anything pending
    python -m rebalancer.migrate status       # list what's applied

The connection string comes from --dsn or DATABASE_URL. Each file runs in its own transaction
and is recorded with a checksum, so an edited file that was already applied is refused rather
than silently skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

import psycopg

_FILE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")
# Any constant works; it only has to be the same for every runner.
_LOCK_ID = 7_243_117


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


def available() -> list[Migration]:
    found = []
    for entry in resources.files("rebalancer").joinpath("migrations").iterdir():
        match = _FILE.match(entry.name)
        if match:
            found.append(Migration(int(match[1]), match[2], entry.read_text()))
    found.sort(key=lambda m: m.version)
    expected = list(range(1, len(found) + 1))
    if [m.version for m in found] != expected:
        raise MigrationError(
            f"migrations must be numbered 001 upwards without gaps, found {[m.version for m in found]}"
        )
    return found


def migrate(conn: psycopg.Connection, migrations: list[Migration] | None = None) -> list[Migration]:
    """Apply pending migrations and return them. The connection must be in autocommit mode."""
    if not conn.autocommit:
        raise MigrationError("open the connection with autocommit=True")
    migrations = available() if migrations is None else migrations
    conn.execute(
        """
        create table if not exists schema_migrations (
            version     integer primary key,
            name        text not null,
            checksum    text not null,
            applied_at  timestamptz not null default now()
        )
        """
    )
    # Two runners starting together would otherwise both see a file as pending.
    conn.execute("select pg_advisory_lock(%s)", (_LOCK_ID,))
    try:
        applied = dict(conn.execute("select version, checksum from schema_migrations").fetchall())
        done = []
        for m in migrations:
            if m.version in applied:
                if applied[m.version] != m.checksum:
                    raise MigrationError(
                        f"{m.version:03d}_{m.name}.sql changed after it was applied; add a new migration instead"
                    )
                continue
            with conn.transaction():
                conn.execute(m.sql)
                conn.execute(
                    "insert into schema_migrations (version, name, checksum) values (%s, %s, %s)",
                    (m.version, m.name, m.checksum),
                )
            done.append(m)
        return done
    finally:
        conn.execute("select pg_advisory_unlock(%s)", (_LOCK_ID,))


def status(conn: psycopg.Connection) -> list[tuple[Migration, datetime | None]]:
    exists = conn.execute("select to_regclass('schema_migrations') is not null").fetchone()[0]
    applied = {}
    if exists:
        applied = dict(conn.execute("select version, applied_at from schema_migrations").fetchall())
    return [(m, applied.get(m.version)) for m in available()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", nargs="?", choices=["up", "status"], default="up")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args(argv)
    if not args.dsn:
        sys.exit("set DATABASE_URL or pass --dsn")

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        if args.command == "status":
            for m, applied_at in status(conn):
                when = applied_at.isoformat(timespec="seconds") if applied_at else "pending"
                print(f"{m.version:03d}_{m.name}  {when}")
            return
        done = migrate(conn)
        if done:
            for m in done:
                print(f"applied {m.version:03d}_{m.name}")
        else:
            print("nothing to apply")


if __name__ == "__main__":
    main()
