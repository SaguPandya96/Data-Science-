"""Causal feature engineering and next-day target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily technical features using information available by the cutoff."""
    output = frame.copy().sort_values("date")
    close = output["close"]
    output["daily_return"] = close.pct_change()
    output["log_return"] = np.log(close).diff()

    for lag in [1, 2, 3, 7]:
        output[f"return_lag_{lag}"] = output["daily_return"].shift(lag)

    for window in [7, 14, 30]:
        historical_returns = output["daily_return"].shift(1).rolling(window)
        output[f"rolling_mean_{window}"] = historical_returns.mean()
        output[f"rolling_volatility_{window}"] = historical_returns.std()
        historical_average = close.shift(1).rolling(window).mean()
        output[f"ma_ratio_{window}"] = close / historical_average - 1

    output["volume_change"] = output["volume"].pct_change()
    historical_volume = output["volume"].shift(1).rolling(30)
    output["volume_zscore_30"] = (
        output["volume"] - historical_volume.mean()
    ) / historical_volume.std()
    output["momentum_7"] = close / close.shift(7) - 1
    output["momentum_30"] = close / close.shift(30) - 1

    delta = close.diff()
    average_gain = delta.clip(lower=0).shift(1).rolling(14).mean()
    average_loss = -delta.clip(upper=0).shift(1).rolling(14).mean()
    relative_strength = average_gain / average_loss
    output["rsi_14"] = 100 - 100 / (1 + relative_strength)

    ema_12 = close.shift(1).ewm(span=12, adjust=False).mean()
    ema_26 = close.shift(1).ewm(span=26, adjust=False).mean()
    output["macd"] = ema_12 - ema_26

    moving_average_20 = close.shift(1).rolling(20).mean()
    standard_deviation_20 = close.shift(1).rolling(20).std()
    output["bollinger_position"] = (
        close - moving_average_20
    ) / (2 * standard_deviation_20)
    output["high_low_range"] = (
        output["high"] - output["low"]
    ) / output["open"]
    output["open_close_spread"] = (
        output["close"] - output["open"]
    ) / output["open"]
    output["rolling_skew_30"] = (
        output["daily_return"].shift(1).rolling(30).skew()
    )
    output["rolling_kurtosis_30"] = (
        output["daily_return"].shift(1).rolling(30).kurt()
    )
    output["drawdown"] = close / close.cummax() - 1
    historical_volatility_median = (
        output["rolling_volatility_30"].expanding().median().shift(1)
    )
    output["volatility_regime"] = (
        output["rolling_volatility_30"] > historical_volatility_median
    ).astype(int)

    output["previous_direction"] = (output["daily_return"] > 0).astype(int)
    output["ma_signal"] = (
        close.shift(1) > close.shift(1).rolling(20).mean()
    ).astype(int)
    output["next_day_return"] = close.shift(-1) / close - 1
    output["target"] = (output["next_day_return"] > 0).astype("Int64")
    output.loc[output["next_day_return"].isna(), "target"] = pd.NA
    return output


def merge_point_in_time(
    market: pd.DataFrame, daily_news: pd.DataFrame
) -> pd.DataFrame:
    """Join one-day-lagged news aggregates to daily market observations."""
    market_frame = market.copy()
    news_frame = daily_news.copy()
    market_frame["date"] = pd.to_datetime(market_frame["date"], utc=True)
    news_frame["date"] = pd.to_datetime(news_frame["date"], utc=True)

    alternative_columns = [column for column in news_frame if column != "date"]
    news_frame[alternative_columns] = news_frame[alternative_columns].shift(1)
    output = market_frame.merge(news_frame, on="date", how="left")

    count_columns = [
        column
        for column in alternative_columns
        if "count" in column or "volume" in column
    ]
    output[count_columns] = output[count_columns].fillna(0)
    output["feature_timestamp"] = output["date"] + pd.Timedelta(
        hours=23, minutes=59
    )
    output["target_timestamp"] = output["date"] + pd.Timedelta(
        days=1, hours=23, minutes=59
    )
    return output
