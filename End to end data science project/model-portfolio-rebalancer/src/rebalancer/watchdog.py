"""Dead-man's switch for the engine, run as its own process.

    export DATABASE_URL=... APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
    export WATCHDOG_WEBHOOK_URL=https://ntfy.sh/<topic>    # optional, for alerts on my phone
    python -m rebalancer.watchdog --account paper-main

Every 30 seconds it:

* reads the engine's heartbeat (its newest cycle). After five minutes without one it cancels
  every order resting at the broker, and keeps doing so until the heartbeat comes back.
* halts trading if the engine's latest reconciliation failed and nothing is halting it.
* sends new alerts on and marks them delivered.

It doesn't halt when the heartbeat stops. Stopping the runner on purpose stops the heartbeat
too, and a restarted engine should pick up where it left off, not wait for me to clear a halt.
Connect it as a login in the rebalancer_watchdog role: that can read everything but only
write halts, alerts and when an alert was delivered.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .venue import Venue

log = logging.getLogger("rebalancer.watchdog")


@dataclass(frozen=True)
class WatchdogConfig:
    heartbeat_timeout: timedelta = timedelta(minutes=5)
    interval: timedelta = timedelta(seconds=30)


@dataclass(frozen=True)
class AlertRow:
    id: int
    ts: datetime
    level: str
    message: str


@dataclass(frozen=True)
class ReconciliationRow:
    id: int
    ts: datetime
    ok: bool
    gap_usd: float | None  # None when the gap was in something the engine couldn't price


class WatchdogStore(Protocol):
    def last_heartbeat(self) -> datetime | None: ...

    def last_reconciliation(self) -> ReconciliationRow | None: ...

    def global_halt_open(self) -> bool: ...

    def halt(self, ts: datetime, reason: str) -> None: ...

    def alert(self, ts: datetime, level: str, message: str) -> None: ...

    def undelivered_alerts(self, since: datetime) -> list[AlertRow]: ...

    def mark_delivered(self, alert_id: int, ts: datetime) -> None: ...


class Notifier(Protocol):
    def send(self, level: str, message: str) -> None:
        """Deliver one alert, raising if it didn't go out."""
        ...


class LogNotifier:
    def send(self, level: str, message: str) -> None:
        log.log(
            logging.INFO if level == "info" else logging.WARNING, "alert (%s): %s", level, message
        )


class WebhookNotifier:
    """POSTs each alert as plain text. The Title and Priority headers are what ntfy reads;
    other services ignore them."""

    PRIORITY = {"info": "default", "warning": "high", "critical": "urgent"}

    def __init__(self, url: str, client=None, timeout: float = 10.0):
        import httpx

        self.url = url
        self._http = client or httpx.Client(timeout=timeout)

    def send(self, level: str, message: str) -> None:
        response = self._http.post(
            self.url,
            content=message.encode(),
            headers={
                "Title": f"rebalancer {level}",
                "Priority": self.PRIORITY.get(level, "default"),
            },
        )
        response.raise_for_status()


class Watchdog:
    # An alert backlog older than this isn't worth waking me for.
    DELIVERY_WINDOW = timedelta(hours=24)

    def __init__(
        self,
        store: WatchdogStore,
        venue: Venue,
        notifier: Notifier,
        *,
        config: WatchdogConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        wait: Callable[[float], None] | None = None,
    ):
        self.store = store
        self.venue = venue
        self.notifier = notifier
        self.config = config or WatchdogConfig()
        self.clock = clock
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self.last_beat: datetime | None = None
        self.engine_silent = False
        self.cancel_error: str | None = None
        self.store_error: str | None = None
        self.halted_for: int | None = None  # the reconciliation the watchdog last halted over

    def stop(self) -> None:
        self._stop.set()

    def run(self, max_checks: int | None = None) -> int:
        checks = 0
        log.info("watching, every %ss", self.config.interval.total_seconds())
        while not self._stop.is_set():
            self.check(self.clock())
            checks += 1
            if max_checks is not None and checks >= max_checks:
                break
            self._wait(self.config.interval.total_seconds())
        return checks

    def check(self, now: datetime) -> None:
        try:
            beat = self.store.last_heartbeat()
        except Exception as exc:
            self._store_failed(now, exc)
        else:
            if self.store_error:
                log.info("database reachable again")
                self.store_error = None
            self.last_beat = beat
        # This runs even when the database is down, from the last heartbeat seen. Not being able
        # to tell whether the engine is alive is treated the same as it being dead.
        self._check_heartbeat(now)
        if self.store_error:
            return
        try:
            self._check_reconciliation(now)
            self._deliver(now)
        except Exception as exc:
            self._store_failed(now, exc)

    # --- checks ----------------------------------------------------------------------------

    def _check_heartbeat(self, now: datetime) -> None:
        if self.last_beat is None:
            return  # the engine has never run on this account
        silent_for = now - self.last_beat
        if silent_for <= self.config.heartbeat_timeout:
            if self.engine_silent:
                self.engine_silent = False
                self._raise(now, "info", "engine heartbeat is back")
            return
        try:
            cancelled = self._cancel_everything()
        except Exception as exc:
            log.exception("could not cancel orders at the broker")
            if self.cancel_error is None:
                self._raise(
                    now,
                    "critical",
                    f"no engine heartbeat for {_minutes(silent_for)} and cancelling orders at "
                    f"the broker failed: {type(exc).__name__}: {exc}",
                )
            self.cancel_error = str(exc)
            return
        if not self.engine_silent:
            self._raise(
                now,
                "critical",
                f"no engine heartbeat since {self.last_beat.isoformat(timespec='seconds')} "
                f"({_minutes(silent_for)}): cancelled {_orders(cancelled)} resting at the broker",
            )
        elif cancelled:
            self._raise(now, "warning", f"engine still silent: cancelled {_orders(cancelled)} more")
        self.engine_silent = True
        self.cancel_error = None

    def _cancel_everything(self) -> int:
        # The account belongs to the engine, so anything resting there is its to cancel.
        cancelled = 0
        for broker_id, client_id in self.venue.open_orders().items():
            if self.venue.cancel(broker_id):
                log.warning("cancelled %s (%s)", client_id, broker_id)
                cancelled += 1
        return cancelled

    def _check_reconciliation(self, now: datetime) -> None:
        # The engine halts itself on a reconciliation gap before it records one. A failed
        # reconciliation with no halt in force means that didn't happen, so something is wrong
        # with the engine as well as the positions.
        rec = self.store.last_reconciliation()
        if rec is None or rec.ok or rec.id == self.halted_for:
            return
        if self.store.global_halt_open():
            return
        gap = "an unpriced amount" if rec.gap_usd is None else f"${rec.gap_usd:,.2f}"
        reason = (
            f"reconciliation at {rec.ts.isoformat(timespec='seconds')} was off by {gap} "
            "and the engine had not halted"
        )
        self.store.halt(now, reason)
        self.halted_for = rec.id
        self._raise(now, "critical", f"watchdog halted trading: {reason}")

    def _deliver(self, now: datetime) -> None:
        for row in self.store.undelivered_alerts(now - self.DELIVERY_WINDOW):
            try:
                self.notifier.send(row.level, row.message)
            except Exception:
                # Stop here so alerts arrive in the order they were raised.
                log.exception("could not deliver alert %s; trying again next check", row.id)
                return
            self.store.mark_delivered(row.id, now)

    # --- internals -------------------------------------------------------------------------

    def _raise(self, now: datetime, level: str, message: str) -> None:
        """Record an alert for the delivery loop to send, or send it straight away if the
        database won't take it."""
        log.log(logging.INFO if level == "info" else logging.WARNING, message)
        if not self.store_error:
            try:
                self.store.alert(now, level, message)
                return
            except Exception as exc:
                self._store_failed(now, exc)
        self._send_now(level, message)

    def _store_failed(self, now: datetime, exc: Exception) -> None:
        log.exception("database error")
        first = self.store_error is None
        self.store_error = f"{type(exc).__name__}: {exc}"
        if first:
            self._send_now("critical", f"watchdog cannot use the database: {self.store_error}")

    def _send_now(self, level: str, message: str) -> None:
        try:
            self.notifier.send(level, message)
        except Exception:
            log.exception("could not send alert: %s", message)


def _minutes(delta: timedelta) -> str:
    return f"{delta.total_seconds() / 60:.0f} min"


def _orders(count: int) -> str:
    return f"{count} order" + ("" if count == 1 else "s")


# --- Postgres ----------------------------------------------------------------------------------


class PostgresWatchdogStore:
    """Reads and writes for the watchdog. It reconnects after losing the database, since the
    watchdog has to outlast whatever took the database away."""

    def __init__(self, connect: Callable[[], object], account_name: str):
        self._connect = connect
        self.conn = None
        row = (
            self._db()
            .execute("select id from accounts where name = %s", (account_name,))
            .fetchone()
        )
        if row is None:
            raise LookupError(f"no account named {account_name}: run the engine on it first")
        self.account_id = row[0]

    def _db(self):
        if self.conn is None or self.conn.closed:
            self.conn = self._connect()
            if not self.conn.autocommit:
                raise ValueError("connect with autocommit=True")
        return self.conn

    def last_heartbeat(self) -> datetime | None:
        return (
            self._db()
            .execute("select max(ts) from cycles where account_id = %s", (self.account_id,))
            .fetchone()[0]
        )

    def last_reconciliation(self) -> ReconciliationRow | None:
        row = (
            self._db()
            .execute(
                """select id, ts, ok, gap_usd from reconciliations where account_id = %s
               order by ts desc, id desc limit 1""",
                (self.account_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        rec_id, ts, ok, gap = row
        return ReconciliationRow(rec_id, ts, ok, None if gap is None else float(gap))

    def global_halt_open(self) -> bool:
        return (
            self._db()
            .execute(
                """select exists (select from halts
                              where (account_id = %s or account_id is null)
                                and scope = 'global' and cleared_at is null)""",
                (self.account_id,),
            )
            .fetchone()[0]
        )

    def halt(self, ts: datetime, reason: str) -> None:
        self._db().execute(
            """insert into halts (account_id, scope, reason, source, started_at)
               values (%s, 'global', %s, 'watchdog', %s)""",
            (self.account_id, reason, ts),
        )

    def alert(self, ts: datetime, level: str, message: str) -> None:
        self._db().execute(
            "insert into alerts (account_id, ts, level, message) values (%s, %s, %s, %s)",
            (self.account_id, ts, level, message),
        )

    def undelivered_alerts(self, since: datetime) -> list[AlertRow]:
        rows = (
            self._db()
            .execute(
                """select id, ts, level, message from alerts
               where (account_id = %s or account_id is null)
                 and delivered_at is null and ts >= %s
               order by ts, id limit 50""",
                (self.account_id, since),
            )
            .fetchall()
        )
        return [AlertRow(*row) for row in rows]

    def mark_delivered(self, alert_id: int, ts: datetime) -> None:
        self._db().execute("update alerts set delivered_at = %s where id = %s", (ts, alert_id))


# --- command line ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import psycopg

    from .alpaca import AlpacaConfig, AlpacaError, AlpacaVenue
    from .replay import DEFAULT_INSTRUMENTS
    from .sessions import MarketCalendar

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--account", required=True, help="the account name the runner uses")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--webhook", default=os.environ.get("WATCHDOG_WEBHOOK_URL"))
    parser.add_argument("--interval", type=int, default=30, help="seconds between checks")
    parser.add_argument("--once", action="store_true", help="run a single check and stop")
    args = parser.parse_args(argv)
    if not args.dsn:
        sys.exit("set DATABASE_URL or pass --dsn")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        alpaca_config = AlpacaConfig.from_env()
    except AlpacaError as exc:
        sys.exit(str(exc))
    try:
        store = PostgresWatchdogStore(
            lambda: psycopg.connect(args.dsn, autocommit=True, connect_timeout=10), args.account
        )
    except LookupError as exc:
        sys.exit(str(exc))
    venue = AlpacaVenue(alpaca_config, DEFAULT_INSTRUMENTS, MarketCalendar())
    notifier = WebhookNotifier(args.webhook) if args.webhook else LogNotifier()
    watchdog = Watchdog(
        store,
        venue,
        notifier,
        config=WatchdogConfig(interval=timedelta(seconds=args.interval)),
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: watchdog.stop())
    watchdog.run(max_checks=1 if args.once else None)


if __name__ == "__main__":
    main()
