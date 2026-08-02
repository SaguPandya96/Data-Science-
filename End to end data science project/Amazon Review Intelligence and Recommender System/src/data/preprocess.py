"""Clean and join the raw Amazon Reviews 2023 (All_Beauty) data.

Produces:
  data/processed/reviews.parquet   - one row per review, joined with product metadata
  data/processed/train.parquet     - time-based train split (all but each user's last review)
  data/processed/test.parquet      - each user's single most recent review (held out)

A minimum-interaction filter is applied so the user-item matrix isn't too sparse
for the collaborative-filtering model to learn anything meaningful.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

CATEGORY = "All_Beauty"
MIN_USER_INTERACTIONS = 2
MIN_ITEM_INTERACTIONS = 3


def load_reviews() -> pd.DataFrame:
    path = RAW_DIR / f"{CATEGORY}.jsonl.gz"
    df = pd.read_json(path, lines=True, compression="gzip")
    df = df[
        ["rating", "title", "text", "asin", "parent_asin", "user_id",
         "timestamp", "helpful_vote", "verified_purchase"]
    ].rename(columns={"title": "review_title", "text": "review_text"})
    return df


def load_metadata() -> pd.DataFrame:
    path = RAW_DIR / f"meta_{CATEGORY}.jsonl.gz"
    df = pd.read_json(path, lines=True, compression="gzip")
    df = df[
        ["parent_asin", "title", "main_category", "average_rating",
         "rating_number", "price", "store", "categories"]
    ].rename(columns={"title": "product_title"})
    return df


def clean_and_join(reviews: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating"])
    reviews = reviews.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])
    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="ms")
    reviews["review_text"] = reviews["review_text"].fillna("")
    reviews["review_title"] = reviews["review_title"].fillna("")

    meta = meta.drop_duplicates(subset=["parent_asin"])
    df = reviews.merge(meta, on="parent_asin", how="left")
    df["product_title"] = df["product_title"].fillna("Unknown product")

    # Filter to a "warm" subset: users and items with enough signal to model.
    for _ in range(5):  # a few passes since filtering one side shrinks the other
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()
        keep_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        keep_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        df = df[df["user_id"].isin(keep_users) & df["parent_asin"].isin(keep_items)]

    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_last = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False) == 1
    test = df[is_last]
    train = df[~is_last]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw reviews and metadata...")
    reviews = load_reviews()
    meta = load_metadata()
    print(f"Raw reviews: {len(reviews):,} | Raw products: {len(meta):,}")

    df = clean_and_join(reviews, meta)
    print(f"After cleaning + min-interaction filter: {len(df):,} reviews, "
          f"{df['user_id'].nunique():,} users, {df['parent_asin'].nunique():,} items")

    train, test = time_based_split(df)
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    df.to_parquet(PROCESSED_DIR / "reviews.parquet", index=False)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Saved parquet files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
