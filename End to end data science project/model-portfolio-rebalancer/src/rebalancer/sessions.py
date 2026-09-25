"""US equity session calendar in Eastern time.

Holidays and half-days come from the XNYS calendar in exchange_calendars, with hand overrides for
anything it gets wrong or doesn't know yet. Around each NYSE trading day D the sessions are:

    overnight  20:00 on D-1 -> 04:00 D     (21:00 once the exchange 23/5 day starts)
    pre        04:00 -> 09:30
    regular    09:30 -> close              (16:00, or 13:00 on a half-day)
    post       close -> 20:00              (17:00 on a half-day)

Anything between those is a gap: the 20:00-21:00 pause once 23/5 starts, the weekend, or plain
closed time (holidays, the evening after a half-day). A gap that takes in a Saturday is labelled
weekend, so a long weekend is one frozen stretch. Because an overnight session only exists in front
of a trading day, the night before a holiday is closed too.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
EXCHANGE_23_5_START = date(2026, 12, 6)

PRE_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
POST_CLOSE = time(20, 0)
HALF_DAY_POST_CLOSE = time(17, 0)
OVERNIGHT_OPEN = time(20, 0)
OVERNIGHT_OPEN_23_5 = time(21, 0)


class Session(StrEnum):
    REGULAR = "regular"
    PRE = "pre"
    POST = "post"
    OVERNIGHT = "overnight"
    PAUSE = "pause"
    WEEKEND = "weekend"
    CLOSED = "closed"

    @property
    def equities_open(self) -> bool:
        return self in _OPEN

    @property
    def off_hours(self) -> bool:
        return self is not Session.REGULAR


_OPEN = frozenset({Session.REGULAR, Session.PRE, Session.POST, Session.OVERNIGHT})


@dataclass(frozen=True)
class SessionWindow:
    session: Session
    start: datetime
    end: datetime

    def __contains__(self, ts: datetime) -> bool:
        return self.start <= ts < self.end


class MarketCalendar:
    def __init__(
        self,
        extra_holidays: set[date] | frozenset[date] = frozenset(),
        extra_half_days: set[date] | frozenset[date] = frozenset(),
        exchange_23_5_start: date | None = EXCHANGE_23_5_START,
        first_year: int = 2020,
        last_year: int = 2030,
    ):
        self._xnys = xcals.get_calendar(
            "XNYS", start=f"{first_year}-01-01", end=f"{last_year}-12-31"
        )
        self._range = (date(first_year, 1, 8), date(last_year, 12, 24))
        self.extra_holidays = frozenset(extra_holidays)
        self.extra_half_days = frozenset(extra_half_days)
        self.exchange_23_5_start = exchange_23_5_start
        # Bind a per-instance cache so overrides don't leak between calendars.
        self._segments_for_day = lru_cache(maxsize=4096)(self._build_segments)

    def is_trading_day(self, d: date) -> bool:
        if not self._range[0] <= d <= self._range[1]:
            raise ValueError(f"{d} is outside the loaded calendar range")
        if d in self.extra_holidays:
            return False
        return bool(self._xnys.is_session(d.isoformat()))

    def is_half_day(self, d: date) -> bool:
        if d in self.extra_half_days:
            return True
        close = self._xnys.session_close(d.isoformat()).tz_convert(ET)
        return close.time() < REGULAR_CLOSE

    def session_at(self, ts: datetime) -> SessionWindow:
        ts = _as_utc(ts)
        day = ts.astimezone(ET).date()
        segments = [
            seg
            for d in _days(day - timedelta(days=5), day + timedelta(days=5))
            for seg in self._segments_for_day(d)
        ]
        segments.sort(key=lambda s: s.start)
        prev_end = None
        for seg in segments:
            if ts in seg:
                return seg
            if seg.start > ts:
                if prev_end is None:
                    raise ValueError(f"no session information before {ts}")
                return SessionWindow(self._label_gap(prev_end, seg.start), prev_end, seg.start)
            prev_end = seg.end
        raise ValueError(f"no session information after {ts}")

    def last_regular_close(self, ts: datetime) -> datetime:
        """End of the most recent regular session that finished at or before ts."""
        ts = _as_utc(ts)
        d = ts.astimezone(ET).date()
        for _ in range(15):
            if self.is_trading_day(d):
                regular = next(s for s in self._segments_for_day(d) if s.session is Session.REGULAR)
                if regular.end <= ts:
                    return regular.end
            d -= timedelta(days=1)
        raise ValueError(f"no regular session in the two weeks before {ts}")

    def _build_segments(self, d: date) -> tuple[SessionWindow, ...]:
        if not self.is_trading_day(d):
            return ()
        half = self.is_half_day(d)
        close = time(13, 0) if half else REGULAR_CLOSE
        post_close = HALF_DAY_POST_CLOSE if half else POST_CLOSE
        eve = d - timedelta(days=1)
        overnight_open = OVERNIGHT_OPEN
        if self.exchange_23_5_start is not None and eve >= self.exchange_23_5_start:
            overnight_open = OVERNIGHT_OPEN_23_5
        return (
            SessionWindow(Session.OVERNIGHT, _at(eve, overnight_open), _at(d, PRE_OPEN)),
            SessionWindow(Session.PRE, _at(d, PRE_OPEN), _at(d, REGULAR_OPEN)),
            SessionWindow(Session.REGULAR, _at(d, REGULAR_OPEN), _at(d, close)),
            SessionWindow(Session.POST, _at(d, close), _at(d, post_close)),
        )

    def _label_gap(self, start: datetime, end: datetime) -> Session:
        start_et, end_et = start.astimezone(ET), end.astimezone(ET)
        d = start_et.date()
        while d <= end_et.date():
            if d.weekday() == 5:
                return Session.WEEKEND
            d += timedelta(days=1)
        if (
            start_et.time() == POST_CLOSE
            and end_et.time() == OVERNIGHT_OPEN_23_5
            and start_et.date() == end_et.date()
        ):
            return Session.PAUSE
        return Session.CLOSED


def _at(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(UTC)


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
