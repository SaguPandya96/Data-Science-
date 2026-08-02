import pandas as pd

from src.backtesting import backtest
from src.sentiment import aggregate_daily


def test_sentiment_aggregation_counts_each_class():
    scored = pd.DataFrame(
        {
            "feature_date": [pd.Timestamp("2024-01-01", tz="UTC")] * 2,
            "headline": ["a", "b"],
            "sentiment_label": ["positive", "negative"],
            "compound_sentiment": [0.5, -0.5],
            "positive_probability": [0.7, 0.1],
            "negative_probability": [0.1, 0.8],
        }
    )

    daily = aggregate_daily(scored)

    assert daily.loc[0, "article_volume"] == 2
    assert daily.loc[0, "positive_count"] == 1
    assert daily.loc[0, "negative_count"] == 1


def test_backtest_uses_the_previous_signal():
    output, _ = backtest(
        pd.date_range("2024-01-01", periods=3),
        probabilities=[0.9, 0.1, 0.1],
        realised_returns=[0.1, 0.1, 0.1],
        threshold=0.5,
        cost_bps=0,
        slippage_bps=0,
    )

    assert output["position"].tolist() == [0, 1, 0]
    assert output["strategy_return"].tolist() == [0, 0.1, 0]
