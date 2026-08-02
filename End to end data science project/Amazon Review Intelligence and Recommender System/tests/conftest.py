import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_reviews() -> pd.DataFrame:
    """Small synthetic review set with a predictable structure.

    5 users x 4 items, ratings correlate with item quality so the models have real signal.
    """
    rng = np.random.default_rng(0)
    rows = []
    item_quality = {"I1": 5.0, "I2": 4.0, "I3": 2.0, "I4": 3.0}
    positive_text = "love this product it works great highly recommend"
    negative_text = "terrible waste of money broke immediately do not buy"

    for user_num in range(1, 6):
        for item_id, quality in item_quality.items():
            rating = int(np.clip(round(quality + rng.normal(0, 0.3)), 1, 5))
            rows.append({
                "user_id": f"U{user_num}",
                "parent_asin": item_id,
                "rating": rating,
                "review_title": "",
                "review_text": positive_text if rating >= 4 else negative_text,
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=user_num * 10 + len(rows)),
                "verified_purchase": True,
                "helpful_vote": 0,
                "product_title": f"Product {item_id}",
                "store": f"Brand{item_id}",
            })
    return pd.DataFrame(rows)
