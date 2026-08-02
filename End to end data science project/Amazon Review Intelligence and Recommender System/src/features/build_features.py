"""Build engineered features for the rating-prediction model.

Features per review:
  user_avg_rating   - that user's historical average rating (excluding current review)
  item_avg_rating   - that item's historical average rating (excluding current review)
  review_length     - word count of review_text
  verified_purchase - bool -> int
  days_since_first_review - recency proxy, days since the user's first review
  sentiment_score    - predicted probability of positive sentiment (filled in by sentiment.py)
"""

import pandas as pd


def _expanding_mean_excl_current(s: pd.Series, fallback: float) -> pd.Series:
    """Historical mean of prior rows only.

    The first row of each group has no history, so it falls back to the *global* mean -
    using the group's own mean there would leak the current row's rating into its feature.
    """
    prior_sum = s.cumsum() - s
    prior_count = pd.Series(range(len(s)), index=s.index, dtype="float64")
    return (prior_sum / prior_count.replace(0, float("nan"))).fillna(fallback).astype("float64")


def add_user_item_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(df["rating"].mean())

    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["user_avg_rating"] = df.groupby("user_id")["rating"].transform(
        _expanding_mean_excl_current, global_mean
    )

    df = df.sort_values(["parent_asin", "timestamp"])
    df["item_avg_rating"] = df.groupby("parent_asin")["rating"].transform(
        _expanding_mean_excl_current, global_mean
    )

    return df.sort_index()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_length"] = df["review_text"].str.split().str.len().fillna(0)
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    return df


def add_recency_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_review = df.groupby("user_id")["timestamp"].transform("min")
    df["days_since_first_review"] = (df["timestamp"] - first_review).dt.days
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_user_item_bias_features(df)
    df = add_text_features(df)
    df = add_recency_feature(df)
    return df
