from __future__ import annotations

from datetime import datetime

import pytest

from keel.clock import FixedClock
from keel.db import connect
from keel.memory.store import MemoryStore
from keel.tools.builtin import builtin_tools
from keel.tools.registry import Toolbox, ToolContext

NOW = datetime(2026, 10, 1, 9, 0)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)


@pytest.fixture
def conn():  # type: ignore[no-untyped-def]
    connection = connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def store(conn, clock) -> MemoryStore:  # type: ignore[no-untyped-def]
    return MemoryStore(conn, clock)


@pytest.fixture
def toolbox(conn, clock, store) -> Toolbox:  # type: ignore[no-untyped-def]
    box = Toolbox(ToolContext(conn, clock, store, "test"))
    for tool in builtin_tools():
        box.register(tool)
    return box
