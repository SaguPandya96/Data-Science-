"""TF-IDF + Logistic Regression sentiment classifier trained on review text + rating.

Sentiment label is derived from the star rating:
  negative: rating in {1, 2}
  neutral:  rating == 3
  positive: rating in {4, 5}

Exposes a `sentiment_score` = P(positive) - P(negative), used both as a standalone
sentiment endpoint and as an engineered feature for the rating predictor.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "sentiment_pipeline.joblib"


def rating_to_label(rating: float) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=3, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])


def train(df: pd.DataFrame) -> Pipeline:
    text = (df["review_title"].fillna("") + " " + df["review_text"].fillna("")).str.strip()
    labels = df["rating"].apply(rating_to_label)

    x_train, x_test, y_train, y_test = train_test_split(
        text, labels, test_size=0.15, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    print(classification_report(y_test, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved sentiment model to {MODEL_PATH}")
    return pipeline


def load() -> Pipeline:
    return joblib.load(MODEL_PATH)


def sentiment_score(pipeline: Pipeline, texts: pd.Series | list[str]) -> pd.Series:
    """Returns P(positive) - P(negative), in [-1, 1]."""
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    pos_idx = classes.index("positive")
    neg_idx = classes.index("negative")
    index = texts.index if isinstance(texts, pd.Series) else None
    return pd.Series(proba[:, pos_idx] - proba[:, neg_idx], index=index)


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")
    train(reviews)
