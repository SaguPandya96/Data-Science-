from pathlib import Path

import pandas as pd
import pytest

from offerlift import planner

ROOT = Path(__file__).resolve().parents[1]

READOUT = planner.Readout(
    priority=planner.GroupEffect(share=0.1, lift=0.14, se=0.01),
    rest=planner.GroupEffect(share=0.9, lift=0.07, se=0.004),
    average_lift=0.077,
    average_se=0.003,
)


def test_small_budget_goes_to_priority_group_only():
    result = planner.plan(READOUT, 100_000, 0.05)
    assert result.priority_emails == 5_000
    assert result.other_emails == 0
    assert result.rule_visits == pytest.approx(5_000 * 0.14)
    assert result.random_visits == pytest.approx(5_000 * 0.077)


def test_large_budget_fills_the_rest_at_random():
    result = planner.plan(READOUT, 100_000, 0.3)
    assert result.priority_emails == 10_000
    assert result.other_emails == 20_000
    assert result.rule_visits == pytest.approx(10_000 * 0.14 + 20_000 * 0.07)


def test_intervals_bracket_estimates():
    result = planner.plan(READOUT, 50_000, 0.2)
    assert result.rule_low < result.rule_visits < result.rule_high
    assert result.random_low < result.random_visits < result.random_high


def test_zero_budget_means_no_emails():
    result = planner.plan(READOUT, 10_000, 0)
    assert result.emails == 0 and result.rule_visits == 0 and result.extra_vs_random == 0


@pytest.mark.parametrize("size, share", [(0, 0.1), (1_000, -0.1), (1_000, 1.5)])
def test_invalid_inputs_are_rejected(size, share):
    with pytest.raises(ValueError):
        planner.plan(READOUT, size, share)


def test_committed_readout_matches_the_confirmed_result():
    readout = planner.load_readout(ROOT / "reports/metrics")
    assert 0.09 < readout.priority.share < 0.11
    assert readout.priority.lift > readout.rest.lift
    # At a 10% budget the planner's gain per 1,000 customers should sit inside the
    # confirmation test's interval.
    result = planner.plan(readout, 1_000, 0.1)
    assert 3.0 < result.extra_vs_random < 8.3


def test_select_customers_prefers_both_category_buyers():
    customers = pd.DataFrame(
        {"id": range(10), "mens": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         "womens": [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]}
    )
    chosen = planner.select_customers(customers, 0.4)
    both = (customers["mens"] == 1) & (customers["womens"] == 1)
    assert chosen["send_email"].sum() == 4
    assert chosen.loc[both, "send_email"].all()
    assert list(chosen["id"]) == list(range(10))


def test_select_customers_validates_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        planner.select_customers(pd.DataFrame({"mens": [1]}), 0.5)
    with pytest.raises(ValueError, match="only 0 and 1"):
        planner.select_customers(pd.DataFrame({"mens": [2], "womens": [0]}), 0.5)
