"""Target construction and observed prevalence tests."""

import pandas as pd


def test_target_is_strictly_more_than_seven_days(shipments):
    expected = (
        pd.to_datetime(shipments["actual_delivery_date"])
        - pd.to_datetime(shipments["scheduled_delivery_date"])
    ).dt.days.gt(7).astype("int8")
    pd.testing.assert_series_equal(shipments["severe_delay"], expected, check_names=False)
    assert shipments.loc[shipments["delivery_delay_days"].eq(7), "severe_delay"].eq(0).all()


def test_observed_target_prevalence(shipments):
    assert int(shipments["severe_delay"].sum()) == 509
    assert len(shipments) == 7030
    assert shipments["severe_delay"].mean() == 509 / 7030


def test_threshold_prevalence_is_monotonic(shipments):
    counts = [int(shipments[f"late_gt_{days}_days"].sum()) for days in (0, 3, 7, 14)]
    assert counts == sorted(counts, reverse=True)
    assert counts == [800, 657, 509, 326]
