"""Injectable time, so tests and the benchmark control what "now" means."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now().replace(microsecond=0)


class FixedClock:
    """A clock that only moves when told to."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def set(self, moment: datetime) -> None:
        self._now = moment

    def advance(self, **kwargs: float) -> None:
        self._now += timedelta(**kwargs)
