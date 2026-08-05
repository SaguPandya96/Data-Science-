"""Identifier generation for runs, sessions, turns, events, tool calls and evaluations.

Two modes exist deliberately. ``new_id`` produces a random UUID for interactive use;
``DeterministicIdFactory`` produces a reproducible sequence from a seed so that a demo
run committed to the repository has byte-identical identifiers when regenerated. The
regression fixtures depend on the second mode.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

_PREFIXES = {
    "run": "run",
    "session": "ses",
    "turn": "trn",
    "event": "evt",
    "tool_call": "tc",
    "evaluation": "eva",
    "scenario": "scn",
    "annotation": "ann",
}


def new_id(kind: str) -> str:
    """Return a random identifier of the given kind, e.g. ``run_1f3c...``."""
    prefix = _PREFIXES.get(kind, kind)
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_id(kind: str, *parts: str | int) -> str:
    """Return a deterministic identifier derived from ``parts``.

    The same inputs always yield the same identifier, which is what makes committed
    demonstration artifacts diff-stable across regeneration.
    """
    prefix = _PREFIXES.get(kind, kind)
    digest = hashlib.blake2b(
        "|".join(str(p) for p in parts).encode("utf-8"), digest_size=6
    ).hexdigest()
    return f"{prefix}_{digest}"


@dataclass
class DeterministicIdFactory:
    """Monotonic, reproducible identifier source scoped to a single run.

    Args:
        seed: Namespace for the generated identifiers. Two factories with the same
            seed emit the same sequence.
    """

    seed: str
    _counters: dict[str, int] = field(default_factory=dict)

    def next(self, kind: str) -> str:
        """Return the next identifier of ``kind`` for this factory's seed."""
        index = self._counters.get(kind, 0)
        self._counters[kind] = index + 1
        return stable_id(kind, self.seed, kind, index)

    def reset(self) -> None:
        """Restart every counter, so the factory replays its original sequence."""
        self._counters.clear()
