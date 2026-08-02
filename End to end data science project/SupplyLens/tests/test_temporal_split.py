"""Chronological split tests."""

from supplylens.modeling import expanding_window_splits, temporal_split


def test_temporal_partitions_are_ordered(shipments, config):
    train, validation, test = temporal_split(shipments, config["splits"])
    assert train["prediction_date"].max() < validation["prediction_date"].min()
    assert validation["prediction_date"].max() < test["prediction_date"].min()
    assert (len(train), len(validation), len(test)) == (4681, 870, 1479)
    assert train["severe_delay"].sum() == 267
    assert validation["severe_delay"].sum() == 83
    assert test["severe_delay"].sum() == 159


def test_expanding_windows_never_train_on_future(shipments):
    for train_index, validation_index, year in expanding_window_splits(shipments):
        train_dates = shipments.loc[train_index, "prediction_date"]
        validation_dates = shipments.loc[validation_index, "prediction_date"]
        assert train_dates.dt.year.max() < year
        assert validation_dates.dt.year.nunique() == 1
        assert validation_dates.dt.year.iloc[0] == year
