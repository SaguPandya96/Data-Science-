"""Budget planner: expected extra visits from the targeting rule vs random targeting.

Uses only the committed readout in ``reports/metrics``: the email's effect within
both-category buyers and within everyone else, measured in the randomized test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

Z_95 = 1.959964


@dataclass(frozen=True)
class GroupEffect:
    share: float  # share of the list in this group
    lift: float  # effect of the email on the visit rate
    se: float  # standard error of that effect


@dataclass(frozen=True)
class Readout:
    priority: GroupEffect  # customers who bought both categories
    rest: GroupEffect  # everyone else
    average_lift: float  # effect across the whole list (random targeting)
    average_se: float


def _se_from_interval(low: float, high: float) -> float:
    return (high - low) / (2 * Z_95)


def load_readout(metrics_dir: str | Path) -> Readout:
    metrics = Path(metrics_dir)
    profile = json.loads((metrics / "targeting_profile.json").read_text())
    effects = json.loads((metrics / "effects.json").read_text())
    purchased = {r["level"]: r for r in profile["segment_lifts"] if r["segment"] == "purchased"}
    both = purchased["both"]
    comparison = profile["bought_both_vs_rest"]["visit"]
    rest_n = sum(r["n"] for level, r in purchased.items() if level != "both")
    rest_rows = [r for level, r in purchased.items() if level != "both"]
    # Everyone else's effect comes from the same randomized test; its standard error is the
    # size-weighted combination of the single-category groups.
    rest_se = float(
        np.sqrt(
            sum(
                (r["n"] / rest_n) ** 2 * _se_from_interval(r["ci_low"], r["ci_high"]) ** 2
                for r in rest_rows
            )
        )
    )
    visit = effects["Mens E-Mail"]["visit"]
    return Readout(
        priority=GroupEffect(
            share=both["share"],
            lift=both["lift"],
            se=_se_from_interval(both["ci_low"], both["ci_high"]),
        ),
        rest=GroupEffect(share=1 - both["share"], lift=comparison["lift_outside"], se=rest_se),
        average_lift=visit["absolute_lift"],
        average_se=visit["standard_error"],
    )


@dataclass(frozen=True)
class Plan:
    emails: int
    priority_emails: int
    other_emails: int
    rule_visits: float
    rule_low: float
    rule_high: float
    random_visits: float
    random_low: float
    random_high: float

    @property
    def extra_vs_random(self) -> float:
        return self.rule_visits - self.random_visits


def plan(readout: Readout, list_size: int, budget_share: float) -> Plan:
    """Email both-category buyers first, then fill the budget at random from everyone else."""
    if list_size <= 0:
        raise ValueError("list_size must be positive")
    if not 0 <= budget_share <= 1:
        raise ValueError("budget_share must be between 0 and 1")
    emails = int(round(list_size * budget_share))
    priority_emails = min(emails, int(round(list_size * readout.priority.share)))
    other_emails = emails - priority_emails

    rule = priority_emails * readout.priority.lift + other_emails * readout.rest.lift
    rule_se = float(
        np.hypot(priority_emails * readout.priority.se, other_emails * readout.rest.se)
    )
    random = emails * readout.average_lift
    random_se = emails * readout.average_se
    return Plan(
        emails=emails,
        priority_emails=priority_emails,
        other_emails=other_emails,
        rule_visits=rule,
        rule_low=rule - Z_95 * rule_se,
        rule_high=rule + Z_95 * rule_se,
        random_visits=random,
        random_low=random - Z_95 * random_se,
        random_high=random + Z_95 * random_se,
    )


def select_customers(
    customers: pd.DataFrame, budget_share: float, random_state: int = 0
) -> pd.DataFrame:
    """Mark who gets the email: both-category buyers first, the rest filled at random.

    ``customers`` needs 0/1 ``mens`` and ``womens`` columns (last year's purchases).
    """
    missing = {"mens", "womens"} - set(customers.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in ["mens", "womens"]:
        if not customers[column].isin([0, 1]).all():
            raise ValueError(f"Column {column} must contain only 0 and 1")
    if not 0 <= budget_share <= 1:
        raise ValueError("budget_share must be between 0 and 1")

    out = customers.copy()
    emails = int(round(len(out) * budget_share))
    priority = (out["mens"] == 1) & (out["womens"] == 1)
    rng = np.random.default_rng(random_state)
    # Priority customers first (in random order), then everyone else in random order.
    order = np.lexsort((rng.random(len(out)), ~priority.to_numpy()))
    selected = np.zeros(len(out), dtype=bool)
    selected[order[:emails]] = True
    out["bought_both"] = priority.to_numpy()
    out["send_email"] = selected
    return out
