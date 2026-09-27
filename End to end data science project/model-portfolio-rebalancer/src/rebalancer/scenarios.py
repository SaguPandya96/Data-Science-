"""Scripted price paths for replays when I don't have a recorded feed to hand.

Every instrument has a latent price that moves around the clock. Crypto prints it every step;
equities only print while their market is open, so whatever the latent price did over a weekend
shows up as a gap at the next open, the same way it does in real data. Shocks are added on top.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .sessions import ET, MarketCalendar

STEP = timedelta(minutes=30)

START_PRICES = {"VOO": 560.0, "VXUS": 68.0, "BTC-USD": 95_000.0, "ETH-USD": 3_400.0}
ANNUAL_VOL = {"VOO": 0.16, "VXUS": 0.15, "BTC-USD": 0.55, "ETH-USD": 0.70}
EQUITIES = frozenset({"VOO", "VXUS"})
# News still arrives while equities are shut, just less of it.
CLOSED_VOL_FACTOR = 0.5


@dataclass(frozen=True)
class Shock:
    instrument: str
    start: datetime
    hours: float
    total_return: float

    def log_step(self, ts: datetime, step: timedelta) -> float:
        steps = self.hours * 3600 / step.total_seconds()
        end = self.start + timedelta(hours=self.hours)
        if self.start <= ts < end:
            return math.log1p(self.total_return) / steps
        return 0.0


def _et(*args: int) -> datetime:
    return datetime(*args, tzinfo=ET)


SAMPLE_START = _et(2026, 11, 2, 9, 30)
SAMPLE_END = _et(2026, 12, 18, 20, 0)
SAMPLE_SHOCKS = (
    # Saturday crash in crypto, partly retraced on Sunday.
    Shock("BTC-USD", _et(2026, 11, 14, 1, 0), 10, -0.20),
    Shock("ETH-USD", _et(2026, 11, 14, 1, 0), 10, -0.24),
    Shock("BTC-USD", _et(2026, 11, 15, 6, 0), 12, 0.06),
    Shock("ETH-USD", _et(2026, 11, 15, 6, 0), 12, 0.07),
    # Weekend news that equities can only price when they reopen Sunday night.
    Shock("VOO", _et(2026, 11, 14, 12, 0), 24, -0.03),
    Shock("VXUS", _et(2026, 11, 14, 12, 0), 24, -0.025),
    # Overnight crypto rally after the 23/5 exchange schedule starts.
    Shock("BTC-USD", _et(2026, 12, 9, 22, 0), 6, 0.14),
    Shock("ETH-USD", _et(2026, 12, 9, 22, 0), 6, 0.16),
)


def generate_prices(
    start: datetime = SAMPLE_START,
    end: datetime = SAMPLE_END,
    *,
    shocks: tuple[Shock, ...] = SAMPLE_SHOCKS,
    seed: int = 7,
    calendar: MarketCalendar | None = None,
    start_prices: dict[str, float] | None = None,
    step: timedelta = STEP,
) -> list[tuple[datetime, dict[str, float]]]:
    calendar = calendar or MarketCalendar()
    rng = random.Random(seed)
    prices = dict(start_prices or START_PRICES)
    steps_per_year = timedelta(days=365) / step
    out: list[tuple[datetime, dict[str, float]]] = []
    ts = start
    while ts <= end:
        open_now = calendar.session_at(ts).session.equities_open
        if ts > start:
            equity_factor, crypto_factor = rng.gauss(0, 1), rng.gauss(0, 1)
            crypto_factor = 0.3 * equity_factor + math.sqrt(1 - 0.09) * crypto_factor
            for inst in prices:
                common = equity_factor if inst in EQUITIES else crypto_factor
                z = 0.9 * common + math.sqrt(1 - 0.81) * rng.gauss(0, 1)
                vol = ANNUAL_VOL[inst] / math.sqrt(steps_per_year)
                if inst in EQUITIES and not open_now:
                    vol *= CLOSED_VOL_FACTOR
                move = (
                    vol * z
                    - 0.5 * vol * vol
                    + sum(s.log_step(ts, step) for s in shocks if s.instrument == inst)
                )
                prices[inst] *= math.exp(move)
        visible = {k: round(v, 2) for k, v in prices.items() if k not in EQUITIES or open_now}
        out.append((ts, visible))
        ts += step
    return out
