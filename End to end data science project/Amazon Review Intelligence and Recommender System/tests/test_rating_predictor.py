import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.models.rating_predictor import FEATURE_COLUMNS
from src.models.sentiment import build_pipeline, rating_to_label, sentiment_score


@pytest.mark.parametrize(("rating", "expected"), [
    (1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive"),
])
def test_rating_to_label_buckets(rating, expected):
    assert rating_to_label(rating) == expected


@pytest.fixture
def sentiment_corpus():
    """The real pipeline uses min_df=3, so each term must appear in at least 3 documents
    for the vocabulary to be non-empty."""
    positive = ["love this product works great recommend"] * 5 + [
        "great quality love it works perfectly"
    ] * 5
    negative = ["terrible waste money broke awful"] * 5 + [
        "awful quality terrible broke waste"
    ] * 5
    texts = pd.Series(positive + negative)
    labels = pd.Series(["positive"] * 10 + ["negative"] * 10)
    return texts, labels


def test_sentiment_pipeline_separates_clear_positive_and_negative(sentiment_corpus):
    texts, labels = sentiment_corpus
    pipeline = build_pipeline().fit(texts, labels)

    scores = sentiment_score(pipeline, texts)

    assert scores[:10].mean() > scores[10:].mean()


def test_sentiment_score_is_bounded(sentiment_corpus):
    texts, labels = sentiment_corpus
    pipeline = build_pipeline().fit(texts, labels)

    scores = sentiment_score(pipeline, texts)

    assert scores.between(-1, 1).all()


def test_feature_columns_are_produced_by_the_feature_builder(sample_reviews):
    features = build_features(sample_reviews)
    features["sentiment_score"] = 0.0

    missing = set(FEATURE_COLUMNS) - set(features.columns)
    assert not missing, f"feature builder is missing: {missing}"
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
