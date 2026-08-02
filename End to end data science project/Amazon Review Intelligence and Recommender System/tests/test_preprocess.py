import pandas as pd

from src.data.preprocess import time_based_split
from src.features.build_features import build_features


def test_time_based_split_holds_out_exactly_one_review_per_user(sample_reviews):
    train, test = time_based_split(sample_reviews)

    assert len(test) == sample_reviews["user_id"].nunique()
    assert test["user_id"].is_unique
    assert len(train) + len(test) == len(sample_reviews)


def test_time_based_split_holds_out_the_most_recent_review(sample_reviews):
    train, test = time_based_split(sample_reviews)

    for user_id, group in test.groupby("user_id"):
        latest_train = train[train["user_id"] == user_id]["timestamp"].max()
        assert group["timestamp"].iloc[0] > latest_train


def test_build_features_produces_expected_columns(sample_reviews):
    result = build_features(sample_reviews)

    for column in ["user_avg_rating", "item_avg_rating", "review_length",
                   "verified_purchase", "days_since_first_review"]:
        assert column in result.columns

    assert result["review_length"].gt(0).all()
    assert result["days_since_first_review"].ge(0).all()


def test_user_avg_rating_excludes_current_review(sample_reviews):
    """Guards against target leakage: a user's average must not include the row itself."""
    result = build_features(sample_reviews).sort_values(["user_id", "timestamp"])
    first_rows = result.groupby("user_id").head(1)

    # The first review for a user has no prior history, so it falls back to the global mean.
    global_mean = sample_reviews["rating"].mean()
    assert (first_rows["user_avg_rating"] == pd.Series(global_mean, index=first_rows.index)).all()
