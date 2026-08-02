"""Headline sentiment scoring and daily aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_vader(news: pd.DataFrame) -> pd.DataFrame:
    """Attach VADER class proportions, compound scores, and labels."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError("Install vaderSentiment to score headlines") from exc

    analyser = SentimentIntensityAnalyzer()
    output = news.copy()
    scores = output["headline"].map(analyser.polarity_scores).apply(pd.Series)
    output["negative_probability"] = scores["neg"]
    output["neutral_probability"] = scores["neu"]
    output["positive_probability"] = scores["pos"]
    output["compound_sentiment"] = scores["compound"]
    output["sentiment_label"] = np.select(
        [
            output["compound_sentiment"] >= 0.05,
            output["compound_sentiment"] <= -0.05,
        ],
        ["positive", "negative"],
        default="neutral",
    )
    output["sentiment_model"] = "VADER"
    return output


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse scored headlines into date-level tone and attention features."""
    headlines = scored.copy()
    headlines["positive"] = (headlines["sentiment_label"] == "positive").astype(int)
    headlines["negative"] = (headlines["sentiment_label"] == "negative").astype(int)
    headlines["neutral"] = (headlines["sentiment_label"] == "neutral").astype(int)

    daily = (
        headlines.groupby("feature_date")
        .agg(
            sentiment_mean=("compound_sentiment", "mean"),
            sentiment_median=("compound_sentiment", "median"),
            max_positive_sentiment=("positive_probability", "max"),
            max_negative_sentiment=("negative_probability", "max"),
            positive_count=("positive", "sum"),
            negative_count=("negative", "sum"),
            neutral_count=("neutral", "sum"),
            article_volume=("headline", "size"),
            sentiment_dispersion=("compound_sentiment", "std"),
        )
        .reset_index()
        .rename(columns={"feature_date": "date"})
    )
    daily["positive_news_ratio"] = daily["positive_count"] / daily["article_volume"]
    daily["negative_news_ratio"] = daily["negative_count"] / daily["article_volume"]
    daily["sentiment_momentum"] = daily["sentiment_mean"].diff()
    daily["sentiment_rolling_7"] = daily["sentiment_mean"].rolling(
        7, min_periods=3
    ).mean()
    daily["sentiment_surprise"] = (
        daily["sentiment_mean"] - daily["sentiment_rolling_7"].shift(1)
    )
    prior_article_volume = (
        daily["article_volume"].rolling(30, min_periods=7).mean().shift(1)
    )
    daily["abnormal_news_volume"] = daily["article_volume"] / prior_article_volume
    expanding_tenth_percentile = (
        daily["sentiment_mean"].expanding().quantile(0.10).shift(1)
    )
    daily["extreme_negative_news"] = (
        daily["sentiment_mean"] < expanding_tenth_percentile
    ).astype(int)
    daily["sentiment_volume_interaction"] = (
        daily["sentiment_mean"] * daily["article_volume"]
    )
    return daily


def score_finbert(news: pd.DataFrame) -> pd.DataFrame:
    """Score headlines with FinBERT when the optional transformer stack is installed."""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification", model="ProsusAI/finbert", top_k=None
    )
    result_rows = []
    for result in classifier(
        news["headline"].tolist(), batch_size=16, truncation=True
    ):
        result_rows.append(
            {item["label"].lower(): item["score"] for item in result}
        )

    output = news.copy()
    probabilities = pd.DataFrame(result_rows)
    for label in ["positive", "negative", "neutral"]:
        output[f"{label}_probability"] = probabilities[label]
    output["compound_sentiment"] = (
        output["positive_probability"] - output["negative_probability"]
    )
    output["sentiment_label"] = probabilities.idxmax(axis=1)
    output["sentiment_model"] = "FinBERT"
    return output
