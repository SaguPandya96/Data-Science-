"""Interpretable rating predictor: HistGradientBoostingRegressor over engineered features.

This model exists alongside the collaborative-filtering recommender specifically because
it's explainable (SHAP) - it answers "why would this user rate this item highly?" in terms
of concrete features, which the latent-factor CF model cannot.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rating_predictor.joblib"

FEATURE_COLUMNS = [
    "user_avg_rating",
    "item_avg_rating",
    "review_length",
    "verified_purchase",
    "days_since_first_review",
    "sentiment_score",
]

# Features knowable BEFORE the customer writes a review. `sentiment_score` and
# `review_length` are derived from the review text itself, so they are only available
# once the customer has already decided their rating. Benchmarking both feature sets
# (see `run_feature_ablation`) is what tells us whether this model actually predicts
# behaviour or merely reads back a rating the customer has already expressed.
PRE_REVIEW_FEATURE_COLUMNS = [
    "user_avg_rating",
    "item_avg_rating",
    "verified_purchase",
    "days_since_first_review",
]


def train(df: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, dict]:
    x = df[FEATURE_COLUMNS]
    y = df["rating"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }
    print(f"Rating predictor - RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved rating predictor to {MODEL_PATH}")
    return model, metrics


def load() -> HistGradientBoostingRegressor:
    return joblib.load(MODEL_PATH)


def build_explainer(model: HistGradientBoostingRegressor):
    """TreeExplainer with no background dataset uses the exact tree-path-dependent
    algorithm: contributions reconcile with the prediction to floating-point precision.

    Supplying a background sample instead switches SHAP to an approximate interventional
    method, which trips its own additivity check on this model. (And supplying the row
    being explained as its own background silently returns all-zero contributions.)
    """
    return shap.TreeExplainer(model)


def explain(model: HistGradientBoostingRegressor, x: pd.DataFrame):
    """Returns a shap.Explanation for the given feature rows."""
    return build_explainer(model)(x[FEATURE_COLUMNS])


def run_feature_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """How much of this model's accuracy survives if we remove signals derived from the
    review text? Compared against predicting the global mean, which is the real baseline
    to beat on a ratings distribution this skewed."""
    y_train = train_df["rating"].to_numpy()
    y_test = test_df["rating"].to_numpy()
    global_mean = float(y_train.mean())

    feature_sets = {
        "global mean (baseline)": None,
        "pre-review features only": PRE_REVIEW_FEATURE_COLUMNS,
        "review text only (sentiment)": ["sentiment_score"],
        "all features": FEATURE_COLUMNS,
    }

    rows = []
    for name, columns in feature_sets.items():
        if columns is None:
            preds = np.full_like(y_test, global_mean, dtype=float)
        else:
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
            )
            model.fit(train_df[columns], y_train)
            preds = model.predict(test_df[columns])

        rows.append({
            "feature_set": name,
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "test_mae": round(float(mean_absolute_error(y_test, preds)), 4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    df = pd.read_parquet(processed_dir / "train_features.parquet")
    train(df)
