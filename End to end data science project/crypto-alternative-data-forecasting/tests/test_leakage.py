import pandas as pd

from src.evaluation import moving_block_accuracy_interval
from src.features import merge_point_in_time
from src.models import chronological_splits


def test_timestamp_and_split_ordering():
    market = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=20, tz="UTC")}
    )
    news = pd.DataFrame(
        {"date": market["date"], "sentiment_mean": range(20)}
    )

    merged = merge_point_in_time(market, news)
    train, validation, test = chronological_splits(20)

    assert (merged["feature_timestamp"] < merged["target_timestamp"]).all()
    assert merged.loc[1, "sentiment_mean"] == 0
    assert max(train) < min(validation) < max(validation) < min(test)


def test_target_columns_are_not_in_the_feature_list():
    features = ["return_lag_1", "sentiment_mean"]

    assert "target" not in features
    assert "next_day_return" not in features


def test_block_interval_is_ordered():
    low, high = moving_block_accuracy_interval(
        [0, 1, 1, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
        block_size=2,
        repetitions=100,
    )

    assert 0 <= low <= high <= 1
