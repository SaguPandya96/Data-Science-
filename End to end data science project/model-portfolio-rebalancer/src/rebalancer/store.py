"""Writes the engine's audit trail to Postgres, one account per store.

    store = PostgresStore.open(conn, model, instruments, account=Account("main", "PA123"))
    engine = Engine(..., audit=AuditLog(store=store))

The connection must be in autocommit mode: every write commits on its own, so an order row is
durable before the order goes to the broker. Opening a store registers the model version, the
instruments and the account, and refuses a model file that changed without a version bump.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import psycopg

from .audit import HaltRow, MarkSeen, OrderRecord, PriceSeen, ReconciliationLine, RecoveredState
from .models import Model
from .venue import Fill, InstrumentInfo

# Orders whose notional counts against turnover: live or filled, or cancelled after a fill.
# Orders count against turnover in full while live or once filled; a cancelled one counts only
# for the share that traded, which is how the engine gave the rest back at the time.
_FILLED = """left join (select order_id, sum(qty) as qty from fills group by order_id) f
             on f.order_id = o.id"""
_COUNTED = """(o.status in ('accepted', 'partially_filled', 'filled')
               or (o.status = 'canceled' and f.qty > 0))"""
_COUNTED_NOTIONAL = """sum(case when o.status = 'canceled' then o.notional * f.qty / o.qty
                        else o.notional end)"""

# The engine's record statuses, in the database's words.
_STATUS = {
    "pending": "pending",
    "placed": "accepted",
    "partially_filled": "partially_filled",
    "rejected": "rejected",
    "filled": "filled",
    "canceled": "canceled",
}


class ModelChangedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Account:
    name: str
    broker_account_id: str
    mode: str = "paper"
    tax_treatment: str = "taxable"
    broker: str = "alpaca"


def code_version(path: Path | None = None) -> str:
    """Short commit of the checkout the code runs from, or 'unknown' outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path or Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


class PostgresStore:
    def __init__(
        self, conn: psycopg.Connection, account_id: int, model_version_id: int, engine_build: str
    ):
        if not conn.autocommit:
            raise ValueError("open the connection with autocommit=True")
        self.conn = conn
        self.account_id = account_id
        self.model_version_id = model_version_id
        self.engine_build = engine_build
        self._cycle_id: int | None = None
        self._decision_ids: dict[str, int] = {}
        self._order_ids: dict[str, int] = {}

    @classmethod
    def open(
        cls,
        conn: psycopg.Connection,
        model: Model,
        instruments: dict[str, InstrumentInfo],
        *,
        account: Account,
        engine_build: str | None = None,
        model_commit: str | None = None,
    ) -> PostgresStore:
        build = engine_build or code_version()
        with conn.transaction():
            model_version_id = _register_model(conn, model, model_commit or build)
            _register_instruments(conn, instruments)
            account_id = _register_account(conn, account)
            current = conn.execute(
                """select model_version_id from account_models where account_id = %s
                   order by effective_from desc limit 1""",
                (account_id,),
            ).fetchone()
            if current is None or current[0] != model_version_id:
                conn.execute(
                    """insert into account_models (account_id, model_version_id, effective_from)
                       values (%s, %s, now())""",
                    (account_id, model_version_id),
                )
        return cls(conn, account_id, model_version_id, build)

    # --- AuditStore ------------------------------------------------------------------------

    def cycle(
        self,
        ts: datetime,
        session: str,
        account_value: float | None,
        halted: str | None,
        prices: list[PriceSeen],
        marks: list[MarkSeen],
    ) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            self._cycle_id = cur.execute(
                """insert into cycles (account_id, model_version_id, ts, session, account_value,
                                       halted, engine_build)
                   values (%s, %s, %s, %s, %s, %s, %s) returning id""",
                (
                    self.account_id,
                    self.model_version_id,
                    ts,
                    session,
                    account_value,
                    halted,
                    self.engine_build,
                ),
            ).fetchone()[0]
            cur.executemany(
                "insert into cycle_prices values (%s, %s, %s, %s, %s)",
                [(self._cycle_id, p.instrument, p.mid, p.quote_ts, p.fresh) for p in prices],
            )
            cur.executemany(
                "insert into sleeve_marks values (%s, %s, %s, %s, %s)",
                [(self._cycle_id, m.sleeve, m.value, m.weight, m.stale) for m in marks],
            )
        self._decision_ids.clear()

    def decision(self, ts: datetime, sleeve: str, action: str, detail: str) -> None:
        self._decision_ids[sleeve] = self.conn.execute(
            """insert into decisions (cycle_id, sleeve, action, detail)
               values (%s, %s, %s, %s) returning id""",
            (self._require_cycle(), sleeve, action, detail),
        ).fetchone()[0]

    def order_created(self, record: OrderRecord) -> None:
        status = _STATUS[record.status]
        with self.conn.transaction():
            order_id = self.conn.execute(
                """insert into orders (client_order_id, broker_order_id, account_id, cycle_id,
                       decision_id, sleeve, instrument, side, qty, order_type, time_in_force,
                       limit_price, reference_price, notional, session, off_hours_window, reason,
                       status, reject_reason, created_at, updated_at)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s)
                   returning id""",
                (
                    record.client_id,
                    record.order_id,
                    self.account_id,
                    self._require_cycle(),
                    self._decision_ids.get(record.sleeve),
                    record.sleeve,
                    record.instrument,
                    record.side,
                    record.qty,
                    record.order_type,
                    record.time_in_force,
                    record.limit_price,
                    record.reference_price,
                    record.notional,
                    record.session,
                    record.off_hours_window,
                    record.reason,
                    status,
                    record.reject_reason or None,
                    record.ts,
                    record.ts,
                ),
            ).fetchone()[0]
            self._event(order_id, record.ts, status, record.reject_reason or "created")
        self._order_ids[record.client_id] = order_id

    def order_updated(self, record: OrderRecord, ts: datetime, detail: str) -> None:
        order_id = self._order_ids[record.client_id]
        status = _STATUS[record.status]
        with self.conn.transaction():
            self.conn.execute(
                """update orders
                   set status = %s, broker_order_id = coalesce(%s, broker_order_id),
                       reject_reason = %s, updated_at = %s
                   where id = %s""",
                (status, record.order_id, record.reject_reason or None, ts, order_id),
            )
            self._event(order_id, ts, status, detail)

    def fill(self, fill: Fill, client_id: str | None) -> None:
        # The broker's fill id is unique, so hearing about a fill twice is harmless.
        self.conn.execute(
            """insert into fills (broker_fill_id, order_id, account_id, instrument, side, qty,
                                  price, fee, ts)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               on conflict (broker_fill_id) do nothing""",
            (
                fill.fill_id,
                self._order_ids.get(client_id) if client_id else None,
                self.account_id,
                fill.instrument,
                fill.side.value,
                fill.qty,
                fill.price,
                fill.fee,
                fill.ts,
            ),
        )

    def alert(self, ts: datetime, level: str, message: str) -> None:
        self.conn.execute(
            "insert into alerts (account_id, ts, level, message) values (%s, %s, %s, %s)",
            (self.account_id, ts, level, message),
        )

    def reconciliation(
        self, ts: datetime, gap_usd: float, ok: bool, lines: list[ReconciliationLine]
    ) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            rec_id = cur.execute(
                """insert into reconciliations (account_id, cycle_id, ts, gap_usd, ok)
                   values (%s, %s, %s, %s, %s) returning id""",
                (self.account_id, self._cycle_id, ts, _dollars(gap_usd), ok),
            ).fetchone()[0]
            cur.executemany(
                "insert into reconciliation_lines values (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        rec_id,
                        ln.instrument,
                        ln.engine_qty,
                        ln.broker_qty,
                        ln.price,
                        _dollars(ln.gap_usd),
                    )
                    for ln in lines
                ],
            )

    def halt_started(
        self, ts: datetime, scope: str, lane: str | None, reason: str, until: datetime | None
    ) -> None:
        self.conn.execute(
            """insert into halts (account_id, scope, lane, reason, source, started_at, until)
               values (%s, %s, %s, %s, 'engine', %s, %s)""",
            (self.account_id, scope, lane, reason, ts, until),
        )

    def halt_cleared(self, ts: datetime, scope: str, lane: str | None) -> None:
        self.conn.execute(
            """update halts set cleared_at = %s, cleared_by = 'engine'
               where account_id = %s and scope = %s and lane is not distinct from %s
                 and cleared_at is null""",
            (ts, self.account_id, scope, lane),
        )

    def cash_flow(self, ts: datetime, amount: float) -> None:
        # Live cash flows come from the broker's activity feed with its own ids; ones the engine
        # is told about directly get a local id.
        self.conn.execute(
            """insert into cash_flows (account_id, broker_activity_id, ts, amount, kind)
               values (%s, %s, %s, %s, %s)""",
            (
                self.account_id,
                f"local-{uuid.uuid4()}",
                ts,
                amount,
                "deposit" if amount > 0 else "withdrawal",
            ),
        )

    def off_hours_need(self, window: datetime, sleeve: str, need: float) -> None:
        self.conn.execute(
            """insert into off_hours_needs (account_id, window_start, sleeve, need)
               values (%s, %s, %s, %s)
               on conflict (account_id, window_start, sleeve)
               do update set need = greatest(off_hours_needs.need, excluded.need)""",
            (self.account_id, window, sleeve, need),
        )

    def open_halts(self, now: datetime) -> list[HaltRow]:
        rows = self.conn.execute(
            """select scope::text, lane, reason, until from halts
               where (account_id = %s or account_id is null) and cleared_at is null
                 and (scope <> 'daily' or until > %s)
               order by started_at, id""",
            (self.account_id, now),
        ).fetchall()
        return [HaltRow(*row) for row in rows]

    def recover(self, now: datetime) -> RecoveredState:
        conn, account = self.conn, self.account_id
        state = RecoveredState(halts=self.open_halts(now))
        state.last_cycle = conn.execute(
            "select max(ts) from cycles where account_id = %s", (account,)
        ).fetchone()[0]

        for row in conn.execute(
            """select o.id, o.created_at, o.session, i.venue, o.broker_order_id, o.client_order_id,
                      o.sleeve, o.instrument, o.side::text, o.qty, o.order_type, o.time_in_force,
                      o.limit_price, o.reference_price, o.notional, o.reason, o.status::text,
                      o.off_hours_window, coalesce(f.qty, 0), f.value, coalesce(f.fee, 0)
               from orders o join instruments i on i.symbol = o.instrument
               left join (select order_id, sum(qty) as qty, sum(qty * price) as value,
                                 sum(fee) as fee
                          from fills group by order_id) f on f.order_id = o.id
               where o.account_id = %s and o.status in ('pending', 'accepted', 'partially_filled')
               order by o.id""",
            (account,),
        ):
            (
                order_id,
                created,
                session,
                venue,
                broker_id,
                client_id,
                sleeve,
                instrument,
                side,
                qty,
                order_type,
                tif,
                limit,
                reference,
                notional,
                reason,
                status,
                window,
                filled_qty,
                filled_value,
                fees,
            ) = row
            self._order_ids[client_id] = order_id
            state.open_orders.append(
                OrderRecord(
                    ts=created,
                    session=session,
                    venue=venue,
                    order_id=broker_id,
                    client_id=client_id,
                    sleeve=sleeve,
                    instrument=instrument,
                    side=side,
                    qty=float(qty),
                    order_type=order_type,
                    time_in_force=tif,
                    limit_price=float(limit) if limit is not None else None,
                    reference_price=float(reference),
                    notional=float(notional),
                    reason=reason,
                    status={"accepted": "placed"}.get(status, status),
                    off_hours_window=window,
                    filled_qty=float(filled_qty),
                    fill_price=float(filled_value / filled_qty) if filled_qty else None,
                    fee=float(fees),
                )
            )

        since = now - timedelta(days=8)
        state.daily_turnover = {
            day: float(total)
            for day, total in conn.execute(
                f"""select (o.created_at at time zone 'America/New_York')::date, {_COUNTED_NOTIONAL}
                    from orders o {_FILLED}
                    where o.account_id = %s and o.created_at >= %s and {_COUNTED}
                    group by 1""",
                (account, since),
            )
        }
        for window, sleeve, asset_class, total in conn.execute(
            f"""select o.off_hours_window, o.sleeve, i.asset_class::text, {_COUNTED_NOTIONAL}
                from orders o join instruments i on i.symbol = o.instrument {_FILLED}
                where o.account_id = %s and o.off_hours_window >= %s and {_COUNTED}
                group by 1, 2, 3""",
            (account, since),
        ):
            state.off_hours_turnover[window] = state.off_hours_turnover.get(window, 0.0) + float(
                total
            )
            if asset_class == "equity":
                state.off_hours_done[(window, sleeve)] = float(total)
        state.off_hours_needs = {
            (window, sleeve): float(need)
            for window, sleeve, need in conn.execute(
                """select window_start, sleeve, need from off_hours_needs
                   where account_id = %s and window_start >= %s""",
                (account, since),
            )
        }
        # Fills from around the last cycle may come back from the broker again; these are the
        # ones already recorded, so they aren't applied to their orders twice.
        if state.last_cycle is not None:
            state.seen_fills = {
                fill_id
                for (fill_id,) in conn.execute(
                    "select broker_fill_id from fills where account_id = %s and ts >= %s",
                    (account, state.last_cycle - timedelta(days=1)),
                )
            }
        # Sales from the last ten days cover any proceeds that can still be settling.
        state.recent_sales = [
            (ts, asset_class, float(net))
            for ts, asset_class, net in conn.execute(
                """select f.ts, i.asset_class::text, f.qty * f.price - f.fee
                   from fills f join instruments i on i.symbol = f.instrument
                   where f.account_id = %s and f.side = 'sell' and f.ts >= %s
                   order by f.ts""",
                (account, now - timedelta(days=10)),
            )
        ]
        # A day of fresh prices is enough to rebuild the crypto drawdown check.
        state.prices = [
            (inst, ts, float(mid))
            for inst, ts, mid in conn.execute(
                """select p.instrument, p.quote_ts, p.mid
                   from cycle_prices p join cycles c on c.id = p.cycle_id
                   where c.account_id = %s and p.fresh and c.ts >= %s
                   order by p.quote_ts, p.instrument""",
                (account, now - timedelta(days=1, hours=1)),
            )
        ]
        return state

    # --- internals -------------------------------------------------------------------------

    def _event(self, order_id: int, ts: datetime, status: str, detail: str) -> None:
        self.conn.execute(
            "insert into order_events (order_id, ts, status, detail) values (%s, %s, %s, %s)",
            (order_id, ts, status, detail),
        )

    def _require_cycle(self) -> int:
        if self._cycle_id is None:
            raise RuntimeError("no cycle recorded yet")
        return self._cycle_id


def _dollars(value: float) -> float | None:
    return None if math.isinf(value) else value


def _register_model(conn: psycopg.Connection, model: Model, commit: str) -> int:
    path = Path(model.source)
    if not path.is_file():
        raise ValueError(f"model {model.name} has no source file to record")
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    row = conn.execute(
        "select id, sha256, loaded_at from model_versions where name = %s and version = %s",
        (model.name, model.version),
    ).fetchone()
    if row is not None:
        if row[1] != digest:
            raise ModelChangedError(
                f"{model.name} version {model.version} differs from the copy recorded on "
                f"{row[2]:%Y-%m-%d}; bump the version in {path.name}"
            )
        return row[0]
    return conn.execute(
        """insert into model_versions (name, version, yaml, sha256, git_commit, cash_floor)
           values (%s, %s, %s, %s, %s, %s) returning id""",
        (model.name, model.version, text, digest, commit, model.cash_floor),
    ).fetchone()[0]


def _register_instruments(conn: psycopg.Connection, instruments: dict[str, InstrumentInfo]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """insert into instruments (symbol, broker_symbol, asset_class, venue, lot_size,
                                        fractionable, refreshed_at)
               values (%s, %s, %s, %s, %s, %s, now())
               on conflict (symbol) do update
               set broker_symbol = excluded.broker_symbol, asset_class = excluded.asset_class,
                   venue = excluded.venue, lot_size = excluded.lot_size,
                   fractionable = excluded.fractionable, refreshed_at = excluded.refreshed_at""",
            [
                (
                    symbol,
                    info.broker_symbol or symbol,
                    info.asset_class,
                    info.venue,
                    info.lot_size,
                    info.lot_size < 1,
                )
                for symbol, info in instruments.items()
            ],
        )


def _register_account(conn: psycopg.Connection, account: Account) -> int:
    row = conn.execute(
        "select id from accounts where broker = %s and broker_account_id = %s",
        (account.broker, account.broker_account_id),
    ).fetchone()
    if row is not None:
        return row[0]
    return conn.execute(
        """insert into accounts (name, broker, broker_account_id, mode, tax_treatment)
           values (%s, %s, %s, %s, %s) returning id""",
        (
            account.name,
            account.broker,
            account.broker_account_id,
            account.mode,
            account.tax_treatment,
        ),
    ).fetchone()[0]
