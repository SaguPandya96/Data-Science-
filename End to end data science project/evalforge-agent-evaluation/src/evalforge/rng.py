"""Seeded randomness.

Every stochastic decision in EvalForge — scenario variation, model degradation, fault
injection, latency jitter, bootstrap resampling — draws from a generator seeded on a
description of *what is being decided*, not on global state.

The consequence is that a decision depends only on its own coordinates. Running scenario
87 alone produces the same faults as running it inside a 150-scenario suite, and running
the suite in parallel produces the same result as running it serially. Without that
property, recovery rates and latency percentiles are not comparable between runs.
"""

from __future__ import annotations

import hashlib
import random
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def derive_seed(*parts: str | int) -> int:
    """Derive a stable 64-bit seed from a tuple of coordinates.

    Args:
        *parts: Anything identifying the decision, e.g. run seed, scenario id, turn
            index and an aspect name such as ``"wrong_argument"``.

    Returns:
        A seed in ``[0, 2**63)``, stable across processes, platforms and Python runs.
        ``hash()`` is deliberately not used: it is randomised per process.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63)


def seeded_random(*parts: str | int) -> random.Random:
    """Return a :class:`random.Random` seeded from ``parts``."""
    return random.Random(derive_seed(*parts))


def seeded_numpy(*parts: str | int) -> np.random.Generator:
    """Return a NumPy generator seeded from ``parts``, for bootstrap resampling."""
    return np.random.default_rng(derive_seed(*parts))


def chance(probability: float, *parts: str | int) -> bool:
    """Decide a seeded Bernoulli trial.

    Args:
        probability: Probability of returning ``True``, in ``[0, 1]``.
        *parts: Coordinates identifying this specific decision.

    Returns:
        Whether the event fires. The same coordinates always give the same answer.
    """
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return seeded_random(*parts).random() < probability


def jitter(base: float, ratio: float, *parts: str | int) -> float:
    """Apply seeded multiplicative jitter to ``base``.

    Args:
        base: The unjittered value, typically a latency in milliseconds.
        ratio: Maximum fractional deviation, e.g. ``0.25`` for +/-25%.
        *parts: Coordinates identifying this specific value.

    Returns:
        ``base`` scaled by a factor drawn from ``[1 - ratio, 1 + ratio]``, never negative.
    """
    if ratio <= 0.0:
        return base
    factor = 1.0 + seeded_random(*parts).uniform(-ratio, ratio)
    return max(0.0, base * factor)


def choose(options: list[T], *parts: str | int) -> T:
    """Pick one element of ``options`` deterministically.

    Raises:
        ValueError: If ``options`` is empty.
    """
    if not options:
        raise ValueError("choose() requires a non-empty sequence")
    return seeded_random(*parts).choice(options)


def sample(options: list[T], count: int, *parts: str | int) -> list[T]:
    """Pick ``count`` distinct elements of ``options`` deterministically.

    If ``count`` exceeds the population, the whole population is returned in a
    deterministically shuffled order rather than raising.
    """
    rng = seeded_random(*parts)
    if count >= len(options):
        shuffled = list(options)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(options, count)


def shuffled(options: list[T], *parts: str | int) -> list[T]:
    """Return a deterministically shuffled copy of ``options``."""
    result = list(options)
    seeded_random(*parts).shuffle(result)
    return result
