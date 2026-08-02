"""Small precomputed lookup tables the API needs to build features for a hypothetical
(user, item, draft-review-text) triple at inference time, without recomputing them from
the full training set on every request.
"""

from pathlib import Path

import joblib
import pandas as pd

LOOKUPS_PATH = Path(__file__).resolve().parents[2] / "models" / "feature_lookups.joblib"


def build_lookups(train_df: pd.DataFrame) -> dict:
    user_stats = train_df.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        first_review=("timestamp", "min"),
    )
    item_stats = train_df.groupby("parent_asin").agg(avg_rating=("rating", "mean"))
    item_titles = (
        train_df.drop_duplicates(subset=["parent_asin"])
        .set_index("parent_asin")["product_title"]
        .to_dict()
    )

    return {
        "global_mean": float(train_df["rating"].mean()),
        "user_avg_rating": user_stats["avg_rating"].to_dict(),
        "user_first_review": user_stats["first_review"].to_dict(),
        "item_avg_rating": item_stats["avg_rating"].to_dict(),
        "item_titles": item_titles,
        "max_timestamp": train_df["timestamp"].max(),
    }


def save(lookups: dict) -> None:
    LOOKUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookups, LOOKUPS_PATH)
    print(f"Saved feature lookups to {LOOKUPS_PATH}")


def load() -> dict:
    return joblib.load(LOOKUPS_PATH)
