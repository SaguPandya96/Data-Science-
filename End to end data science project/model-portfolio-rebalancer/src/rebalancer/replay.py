"""Replay a price history through the engine against simulated venues.

    python -m rebalancer.replay                          # all models, scripted sample prices
    python -m rebalancer.replay --model models/growth-247.yaml --prices data/my_feed.csv

A price file is a CSV with columns ts, instrument, price (ts in ISO format with an offset). Rows
only need to exist while an instrument's market is open; the gaps are the point.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

import pandas as pd

from .audit import Alert, AuditLog, Decision, OrderRecord
from .engine import CycleReport, Engine, EngineConfig
from .models import Model, load_model, load_models
from .risk import RiskLimits
from .scenarios import generate_prices
from .sessions import MarketCalendar
from .sim import SimAccount, SimClock, SimVenue
from .venue import InstrumentInfo

PriceSteps = list[tuple[datetime, dict[str, float]]]

DEFAULT_INSTRUMENTS = {
    "VOO": InstrumentInfo("equities", "equity", 1.0),
    "VXUS": InstrumentInfo("equities", "equity", 1.0),
    "BTC-USD": InstrumentInfo("crypto", "crypto", 1e-6),
    "ETH-USD": InstrumentInfo("crypto", "crypto", 1e-5),
}
CRYPTO_FEE_BPS = 10.0


@dataclass
class World:
    model: Model
    clock: SimClock
    account: SimAccount
    venues: dict[str, SimVenue]
    engine: Engine
    instruments: dict[str, InstrumentInfo]

    def set_prices(self, ts: datetime, prices: dict[str, float]) -> None:
        self.clock.now = ts
        for inst, price in prices.items():
            if inst in self.instruments:
                self.venues[self.instruments[inst].venue].set_price(inst, price, ts)

    def step(self, ts: datetime, prices: dict[str, float] | None = None) -> CycleReport:
        self.set_prices(ts, prices or {})
        return self.engine.cycle(ts)

    def deposit(self, ts: datetime, amount: float) -> None:
        self.account.cash += amount
        self.engine.record_cash_flow(ts, amount)


def build_world(
    model: Model,
    start: datetime,
    prices: dict[str, float],
    *,
    value: float = 100_000.0,
    weights: dict[str, float] | None = None,
    instruments: dict[str, InstrumentInfo] | None = None,
    calendar: MarketCalendar | None = None,
    limits: RiskLimits | None = None,
    config: EngineConfig | None = None,
    price_times: dict[str, datetime] | None = None,
) -> World:
    """Fund an account at the model's targets (or the given weights) and start an engine on it."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    calendar = calendar or MarketCalendar()
    weights = weights or {s.id: s.target for s in model.sleeves}
    clock = SimClock(start)
    account = SimAccount(cash=value)
    by_venue: dict[str, set[str]] = {}
    for inst, info in instruments.items():
        by_venue.setdefault(info.venue, set()).add(inst)
    venues = {
        name: SimVenue(
            name,
            insts,
            clock,
            account,
            calendar,
            always_open=all(instruments[i].asset_class == "crypto" for i in insts),
            fee_bps=CRYPTO_FEE_BPS
            if all(instruments[i].asset_class == "crypto" for i in insts)
            else 0.0,
            reports_cash=(name == "equities"),
        )
        for name, insts in by_venue.items()
    }
    price_times = price_times or {}
    for inst, price in prices.items():
        if inst in instruments:
            venues[instruments[inst].venue].set_price(inst, price, price_times.get(inst, start))

    for sleeve in model.invested:
        inst = sleeve.all_instruments[0]
        info = instruments[inst]
        lots = int(value * weights.get(sleeve.id, 0.0) / prices[inst] / info.lot_size)
        qty = round(lots * info.lot_size, 10)
        account.holdings[inst] = qty
        account.cash -= qty * prices[inst]

    audit = AuditLog()
    engine = Engine(model, venues, instruments, calendar, limits=limits, config=config, audit=audit)
    engine.start(start)
    return World(model, clock, account, venues, engine, instruments)


@dataclass
class ReplayResult:
    model: Model
    label: str
    snapshots: pd.DataFrame
    orders: pd.DataFrame
    decisions: pd.DataFrame
    alerts: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_replay(
    model: Model,
    steps: PriceSteps,
    *,
    value: float = 100_000.0,
    config: EngineConfig | None = None,
    limits: RiskLimits | None = None,
    calendar: MarketCalendar | None = None,
    trade: bool = True,
    weights: dict[str, float] | None = None,
    events: dict[datetime, Callable[[World], None]] | None = None,
    label: str = "engine",
) -> ReplayResult:
    calendar = calendar or MarketCalendar()
    needed = {i for s in model.invested for i in s.all_instruments}
    seen: dict[str, tuple[datetime, float]] = {}
    begin = None
    for i, (ts, prices) in enumerate(steps):
        for inst, price in prices.items():
            seen[inst] = (ts, price)
        if needed <= set(seen):
            begin = i
            break
    if begin is None:
        raise ValueError(f"price history never covers all of {sorted(needed)}")

    start = steps[begin][0]
    world = build_world(
        model,
        start,
        {k: p for k, (_, p) in seen.items()},
        value=value,
        weights=weights,
        calendar=calendar,
        limits=limits,
        config=config,
        price_times={k: t for k, (t, _) in seen.items()},
    )
    events = events or {}
    rows = []
    for ts, prices in steps[begin:]:
        if ts in events:
            events[ts](world)
        if trade:
            report = world.step(ts, prices)
        else:
            world.set_prices(ts, prices)
            report = world.engine.snapshot(ts)
        row = {
            "ts": ts,
            "session": report.session.value,
            "value": report.account_value,
            "halted": report.halted or "",
        }
        for sid, mark in report.marks.items():
            row[f"w_{sid}"] = mark.weight
            row[f"stale_{sid}"] = mark.stale
        rows.append(row)

    audit = world.engine.audit
    result = ReplayResult(
        model=model,
        label=label,
        snapshots=pd.DataFrame(rows),
        orders=_frame(audit.rows("orders"), OrderRecord),
        decisions=_frame(audit.rows("decisions"), Decision),
        alerts=_frame(audit.rows("alerts"), Alert),
    )
    result.summary = summarize(result)
    return result


def _frame(rows: list[dict], kind: type) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[f.name for f in fields(kind)])


def summarize(result: ReplayResult) -> dict:
    snaps, orders, model = result.snapshots, result.orders, result.model
    out: dict = {
        "model": model.name,
        "run": result.label,
        "start": snaps["ts"].iloc[0].isoformat(),
        "end": snaps["ts"].iloc[-1].isoformat(),
        "cycles": len(snaps),
        "start_value": round(float(snaps["value"].iloc[0]), 2),
        "end_value": round(float(snaps["value"].iloc[-1]), 2),
    }
    sleeves = {}
    for sleeve in model.sleeves:
        w = snaps[f"w_{sleeve.id}"]
        info = {
            "target": sleeve.target,
            "max_abs_drift_pp": round(float((w - sleeve.target).abs().max() * 100), 2),
        }
        edges = sleeve.band_edges()
        if edges is not None:
            outside = (w < edges[0] - 1e-9) | (w > edges[1] + 1e-9)
            info["share_of_time_outside_band"] = round(float(outside.mean()), 4)
        sleeves[sleeve.id] = info
    out["sleeves"] = sleeves
    out["min_cash_weight"] = round(float(snaps[f"w_{model.cash.id}"].min()), 4)

    if orders.empty:
        out["orders"] = {"placed": 0, "filled": 0, "rejected": 0}
        out["filled_turnover_pct_of_start"] = 0.0
        out["fees"] = 0.0
    else:
        filled = orders[orders["status"] == "filled"]
        out["orders"] = {
            "placed": int((orders["status"] != "rejected").sum()),
            "filled": int(len(filled)),
            "rejected": int((orders["status"] == "rejected").sum()),
            "filled_by_session": {
                k: int(v) for k, v in filled["session"].value_counts().sort_index().items()
            },
        }
        traded = (filled["qty"] * filled["fill_price"]).sum()
        out["filled_turnover_pct_of_start"] = round(float(traded / out["start_value"] * 100), 2)
        out["fees"] = round(float(filled["fee"].sum()), 2)
    out["alerts"] = int(len(result.alerts))
    out["halted_cycles"] = int((snaps["halted"] != "").sum())
    return out


def load_price_csv(path: str | Path) -> PriceSteps:
    grouped: dict[datetime, dict[str, float]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            ts = datetime.fromisoformat(row["ts"])
            if ts.tzinfo is None:
                raise ValueError(f"{path}: timestamp {row['ts']} has no UTC offset")
            grouped.setdefault(ts, {})[row["instrument"]] = float(row["price"])
    return sorted(grouped.items())


def write_result(result: ReplayResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("orders", "decisions", "alerts"):
        frame = getattr(result, name)
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")


def comparison_table(results: Iterable[ReplayResult]) -> str:
    lines = [
        "| Model | Run | End value | Worst sleeve drift (pp) | Time any sleeve out of band | Orders filled | Turnover (% of start) | Fees |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s = r.summary
        worst = max(v["max_abs_drift_pp"] for k, v in s["sleeves"].items() if k != r.model.cash.id)
        snaps = r.snapshots
        out_any = pd.Series(False, index=snaps.index)
        for sleeve in r.model.invested:
            lo, hi = sleeve.band_edges()
            w = snaps[f"w_{sleeve.id}"]
            out_any |= (w < lo - 1e-9) | (w > hi + 1e-9)
        lines.append(
            f"| {r.model.name} | {r.label} | ${s['end_value']:,.0f} | {worst:.2f} | {out_any.mean():.1%} | "
            f"{s['orders']['filled']} | {s['filled_turnover_pct_of_start']:.1f}% | ${s['fees']:,.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", type=Path, help="one model file (default: every file in models/)"
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--prices", type=Path, help="price CSV (default: the scripted sample)")
    parser.add_argument("--value", type=float, default=100_000.0)
    parser.add_argument("--out", type=Path, default=Path("reports/sample_replay"))
    args = parser.parse_args(argv)

    models = [load_model(args.model)] if args.model else list(load_models(args.models_dir).values())
    steps = load_price_csv(args.prices) if args.prices else generate_prices()
    calendar = MarketCalendar()
    results = []
    for model in models:
        runs = [
            run_replay(model, steps, value=args.value, calendar=calendar, label="24/7 engine"),
            run_replay(
                model,
                steps,
                value=args.value,
                calendar=calendar,
                label="regular hours only",
                config=EngineConfig(regular_hours_only=True),
            ),
            run_replay(
                model,
                steps,
                value=args.value,
                calendar=calendar,
                label="no rebalancing",
                trade=False,
            ),
        ]
        write_result(runs[0], args.out / model.name)
        results.extend(runs)
    (args.out / "comparison.md").write_text(comparison_table(results))
    print(comparison_table(results))
    print(f"orders, decisions and alerts written under {args.out}/")


if __name__ == "__main__":
    main()
