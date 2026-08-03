"""Tests for threshold selection and rate estimation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evasion_gap.metrics import (  # noqa: E402
    bootstrap_ci,
    rate_above,
    threshold_at_fpr,
    threshold_at_recall,
)

rng = np.random.default_rng(0)
TOXIC = rng.beta(5, 2, size=500)
BENIGN = rng.beta(2, 5, size=500)


@pytest.mark.parametrize("target", [0.80, 0.90, 0.95])
def test_threshold_at_recall_hits_target(target):
    t = threshold_at_recall(TOXIC, target)
    assert rate_above(TOXIC, t) == pytest.approx(target, abs=0.02)


@pytest.mark.parametrize("target", [0.01, 0.05, 0.10])
def test_threshold_at_fpr_respects_budget(target):
    t = threshold_at_fpr(BENIGN, target)
    assert rate_above(BENIGN, t) <= target + 0.02


def test_higher_recall_target_gives_lower_threshold():
    assert threshold_at_recall(TOXIC, 0.99) < threshold_at_recall(TOXIC, 0.50)


def test_tighter_fpr_budget_gives_higher_threshold():
    assert threshold_at_fpr(BENIGN, 0.01) > threshold_at_fpr(BENIGN, 0.10)


def test_rate_above_bounds():
    assert rate_above(TOXIC, -1.0) == 1.0
    assert rate_above(TOXIC, 2.0) == 0.0


def test_bootstrap_ci_contains_point_estimate():
    t = threshold_at_recall(TOXIC, 0.90)
    lo, hi = bootstrap_ci(TOXIC, t)
    assert lo <= rate_above(TOXIC, t) <= hi


def test_bootstrap_ci_is_deterministic():
    t = threshold_at_recall(TOXIC, 0.90)
    assert bootstrap_ci(TOXIC, t, seed=1) == bootstrap_ci(TOXIC, t, seed=1)
