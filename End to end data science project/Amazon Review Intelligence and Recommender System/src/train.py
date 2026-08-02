"""End-to-end training pipeline entry point.

Run order matters here to avoid leakage:
  1. Build engineered features on the TRAIN split only (expanding/historical means).
  2. Train the sentiment model on TRAIN review text.
  3. Score sentiment on TRAIN and TEST (predict only - never re-fit on test).
  4. Derive TEST's user/item average-rating features from TRAIN aggregates (since in a
     real deployment you'd only know a user's/item's history up to "now").
  5. Train the rating predictor (HistGradientBoosting) on TRAIN features, evaluate on TEST.
  6. Train the hybrid recommender (matrix factorization + content-based) on TRAIN.
  7. Evaluate ranking quality (Hit Rate@K, NDCG@K) via leave-one-out on TEST.
  8. Save feature lookups the API needs to serve requests without recomputing from raw data.
"""

import json
from pathlib import Path

import pandas as pd

from src.features import build_features, lookups
from src.models import evaluate, rating_predictor, recommender, sentiment

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    print("Loading train/test splits...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("\n=== Building train features ===")
    train_features = build_features.build_features(train_df)

    print("\n=== Training sentiment model ===")
    sentiment_pipeline = sentiment.train(train_features)
    train_features["sentiment_score"] = sentiment.sentiment_score(
        sentiment_pipeline,
        (train_features["review_title"].fillna("") + " " + train_features["review_text"].fillna("")),
    ).to_numpy()

    print("\n=== Building test features (from train-derived aggregates, no leakage) ===")
    user_avg = train_df.groupby("user_id")["rating"].mean()
    item_avg = train_df.groupby("parent_asin")["rating"].mean()
    user_first_review = train_df.groupby("user_id")["timestamp"].min()
    global_mean = train_df["rating"].mean()

    test_features = test_df.copy()
    test_features["user_avg_rating"] = test_features["user_id"].map(user_avg).fillna(global_mean)
    test_features["item_avg_rating"] = test_features["parent_asin"].map(item_avg).fillna(global_mean)
    test_features["review_length"] = test_features["review_text"].str.split().str.len().fillna(0)
    test_features["verified_purchase"] = test_features["verified_purchase"].astype(int)
    first_review_mapped = test_features["user_id"].map(user_first_review)
    test_features["days_since_first_review"] = (
        (test_features["timestamp"] - first_review_mapped).dt.days.fillna(0)
    )
    test_features["sentiment_score"] = sentiment.sentiment_score(
        sentiment_pipeline,
        (test_features["review_title"].fillna("") + " " + test_features["review_text"].fillna("")),
    ).to_numpy()

    train_features.to_parquet(PROCESSED_DIR / "train_features.parquet", index=False)
    test_features.to_parquet(PROCESSED_DIR / "test_features.parquet", index=False)

    print("\n=== Training rating predictor (HistGradientBoosting + SHAP-ready) ===")
    model, train_metrics = rating_predictor.train(train_features)
    test_preds = model.predict(test_features[rating_predictor.FEATURE_COLUMNS])
    test_metrics = evaluate.rating_rmse_mae(test_features["rating"].to_numpy(), test_preds)
    print(f"Rating predictor - held-out TEST RMSE: {test_metrics['rmse']:.3f}, "
          f"MAE: {test_metrics['mae']:.3f}")

    print("\n=== Training hybrid recommender (matrix factorization + content-based) ===")
    hybrid = recommender.build_and_train(train_df)

    print("\n=== Feature ablation: does this model predict, or just read back the review? ===")
    ablation = rating_predictor.run_feature_ablation(train_features, test_features)
    print(ablation.to_string(index=False))

    print("\n=== Benchmarking rankers against baselines ===")
    comparison = evaluate.compare_rankers(hybrid, train_df, test_df, k=10, max_users=2000)
    print(comparison.to_string(index=False))

    print("\n=== Ranking quality by user activity level ===")
    by_segment = evaluate.evaluate_by_user_activity(hybrid, train_df, test_df, k=10)
    print(by_segment.to_string(index=False))

    print("\n=== Saving feature lookups for the API ===")
    lookup_table = lookups.build_lookups(train_df)
    lookups.save(lookup_table)

    metrics_out = {
        "dataset": {
            "modeled_reviews": int(len(train_df) + len(test_df)),
            "users": int(train_df["user_id"].nunique()),
            "items": int(train_df["parent_asin"].nunique()),
            "mean_train_interactions_per_user": round(
                float(train_df["user_id"].value_counts().mean()), 2
            ),
        },
        "rating_predictor": {"train": train_metrics, "test": test_metrics},
        "feature_ablation": ablation.to_dict(orient="records"),
        "ranker_comparison": comparison.to_dict(orient="records"),
        "by_user_activity": by_segment.to_dict(orient="records"),
    }
    metrics_path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"Saved metrics to {metrics_path}")

    print("\nDone. Summary:")
    print(f"  Rating predictor - TRAIN RMSE {train_metrics['rmse']:.3f} / "
          f"TEST RMSE {test_metrics['rmse']:.3f}")


if __name__ == "__main__":
    main()
