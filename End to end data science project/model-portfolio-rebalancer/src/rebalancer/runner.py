"""Run the engine on a schedule against the Alpaca paper account.

    export DATABASE_URL=... APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
    python -m rebalancer.runner --model models/growth-247.yaml --account paper-main --broker-account PA...

One cycle a minute, on the minute. The runner claims the account in Postgres first, so a second
runner on the same account refuses to start. Ctrl-C or SIGTERM finishes the current cycle,
cancels whatever is resting at the broker, and exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .audit import AuditLog
from .engine import CycleReport, Engine, EngineConfig
from .models import Model, load_model
from .sessions import MarketCalendar
from .venue import InstrumentInfo, Venue

log = logging.getLogger("rebalancer.runner")


@dataclass(frozen=True)
class RunnerConfig:
    interval: timedelta = timedelta(seconds=60)
    # This many failed cycles in a row halts trading. A single failure, like a broker timeout,
    # is logged and the next cycle tries again.
    max_consecutive_errors: int = 3


class Runner:
    def __init__(
        self,
        engine: Engine,
        *,
        config: RunnerConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        wait: Callable[[float], None] | None = None,
    ):
        self.engine = engine
        self.config = config or RunnerConfig()
        self.clock = clock
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self.errors_in_a_row = 0
        self.cycles = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self, max_cycles: int | None = None) -> int:
        """Cycle until stopped (or for max_cycles), then stand down. Returns cycles run."""
        self.engine.start(self.clock())
        log.info(
            "started: %s, one cycle every %ss",
            self.engine.model.name,
            self.config.interval.total_seconds(),
        )
        try:
            while not self._stop.is_set():
                self.tick(self.clock())
                if max_cycles is not None and self.cycles >= max_cycles:
                    break
                self._wait(self._seconds_to_next_tick())
        finally:
            self._stand_down()
        return self.cycles

    def tick(self, now: datetime) -> CycleReport | None:
        self.cycles += 1
        try:
            report = self.engine.cycle(now)
        except Exception as exc:  # a cycle must never take the loop down with it
            self.errors_in_a_row += 1
            message = f"cycle failed ({self.errors_in_a_row} in a row): {type(exc).__name__}: {exc}"
            log.exception(message)
            self.engine.audit.alert(now, "critical", message)
            if self.errors_in_a_row >= self.config.max_consecutive_errors:
                self.engine.gate.halt(
                    now, f"{self.errors_in_a_row} cycles in a row failed, last: {exc}"
                )
            return None
        self.errors_in_a_row = 0
        placed = [o for o in report.orders if o.status != "rejected"]
        log.info(
            "%s %s value=%s orders=%d rejected=%d%s",
            now.isoformat(timespec="seconds"),
            report.session,
            f"{report.account_value:,.2f}" if report.account_value else "-",
            len(placed),
            len(report.orders) - len(placed),
            f" HALTED: {report.halted}" if report.halted else "",
        )
        return report

    def _seconds_to_next_tick(self) -> float:
        # Ticks land on multiples of the interval, so a slow cycle skips ahead rather than
        # running the missed ones back to back.
        now = self.clock()
        step = self.config.interval.total_seconds()
        elapsed = now.timestamp() % step
        return step - elapsed

    def _stand_down(self) -> None:
        now = self.clock()
        try:
            self.engine.stand_down(now)
            log.info("stopped after %d cycles; resting orders cancelled", self.cycles)
        except Exception:
            # The watchdog cancels orders if the heartbeat stops, so this isn't the last line.
            log.exception("could not cancel resting orders on the way out")


# --- command line ------------------------------------------------------------------------------


def build(
    model: Model,
    *,
    dsn: str,
    account_name: str,
    broker_account: str,
    venue_factory: Callable[[dict[str, InstrumentInfo], MarketCalendar], Venue],
    instruments: dict[str, InstrumentInfo],
    calendar: MarketCalendar | None = None,
    config: EngineConfig | None = None,
) -> Engine:
    import psycopg

    from .store import Account, PostgresStore

    calendar = calendar or MarketCalendar()
    conn = psycopg.connect(dsn, autocommit=True)
    store = PostgresStore.open(
        conn, model, instruments, account=Account(account_name, broker_account, mode="paper")
    )
    if not store.try_lock_account():
        conn.close()
        raise SystemExit(f"account {account_name} is already being run by another process")
    venue = venue_factory(instruments, calendar)
    return Engine(
        model,
        {venue.name: venue},
        instruments,
        calendar,
        config=config,
        audit=AuditLog(store=store),
    )


def main(argv: list[str] | None = None) -> None:
    import os

    from .alpaca import AlpacaConfig, AlpacaError, AlpacaVenue
    from .replay import DEFAULT_INSTRUMENTS

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--account", required=True, help="name for this account in the database")
    parser.add_argument("--broker-account", required=True, help="the Alpaca paper account number")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--interval", type=int, default=60, help="seconds between cycles")
    parser.add_argument("--once", action="store_true", help="run a single cycle and stop")
    args = parser.parse_args(argv)
    if not args.dsn:
        sys.exit("set DATABASE_URL or pass --dsn: the runner needs Postgres for its audit trail")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request at INFO, which buries the one line per cycle.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        alpaca_config = AlpacaConfig.from_env()
    except AlpacaError as exc:
        sys.exit(str(exc))
    engine = build(
        load_model(args.model),
        dsn=args.dsn,
        account_name=args.account,
        broker_account=args.broker_account,
        venue_factory=lambda instruments, calendar: AlpacaVenue(
            alpaca_config, instruments, calendar
        ),
        instruments=DEFAULT_INSTRUMENTS,
    )
    runner = Runner(engine, config=RunnerConfig(interval=timedelta(seconds=args.interval)))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.stop())
    runner.run(max_cycles=1 if args.once else None)


if __name__ == "__main__":
    main()
