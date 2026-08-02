"""API tests.

These run against the real trained artifacts, so they're skipped when the models haven't
been built yet (e.g. on a fresh clone or in CI without the dataset). The point is to catch
serving-layer regressions - schema drift, cold-start crashes, unknown-ID handling - not to
re-test model quality.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models.recommender import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model artifacts not found - run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def known_user_and_item(client):
    import pandas as pd

    from src.api.main import PROCESSED_DIR

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["user_id", "parent_asin"])
    return train.iloc[0]["user_id"], train.iloc[0]["parent_asin"]


def test_health_reports_loaded_models(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_returns_requested_number_of_items(client, known_user_and_item):
    user_id, _ = known_user_and_item
    response = client.get(f"/recommend/{user_id}", params={"k": 5})

    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 5
    assert body["strategy"] == "collaborative"


def test_recommend_never_returns_items_the_user_already_reviewed(client, known_user_and_item):
    import pandas as pd

    from src.api.main import PROCESSED_DIR

    user_id, _ = known_user_and_item
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["user_id", "parent_asin"])
    seen = set(train[train["user_id"] == user_id]["parent_asin"])

    body = client.get(f"/recommend/{user_id}", params={"k": 20}).json()
    recommended = {item["parent_asin"] for item in body["recommendations"]}

    assert not recommended & seen


def test_unknown_user_falls_back_instead_of_failing(client):
    response = client.get("/recommend/NOT_A_REAL_USER_ID", params={"k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["is_cold_start"] is True
    assert body["strategy"] == "popularity"
    assert len(body["recommendations"]) > 0


def test_similar_items_returns_other_products(client, known_user_and_item):
    _, parent_asin = known_user_and_item
    response = client.get(f"/similar/{parent_asin}", params={"k": 5})

    assert response.status_code == 200
    body = response.json()
    assert len(body["similar_items"]) == 5
    assert all(item["parent_asin"] != parent_asin for item in body["similar_items"])


def test_similar_items_404s_for_unknown_product(client):
    assert client.get("/similar/NOT_A_REAL_ASIN").status_code == 404


def test_sentiment_distinguishes_positive_from_negative(client):
    positive = client.post("/sentiment", json={"text": "I love this, it works beautifully"}).json()
    negative = client.post("/sentiment", json={"text": "Terrible, broke instantly, waste of money"}).json()

    assert positive["score"] > negative["score"]
    assert positive["label"] == "positive"


def test_sentiment_rejects_empty_text(client):
    assert client.post("/sentiment", json={"text": ""}).status_code == 422


def test_predict_rating_returns_in_range_prediction_with_explanation(client, known_user_and_item):
    user_id, parent_asin = known_user_and_item
    response = client.post("/predict_rating", json={
        "user_id": user_id,
        "parent_asin": parent_asin,
        "review_text": "This product is wonderful, I use it every day",
        "verified_purchase": True,
    })

    assert response.status_code == 200
    body = response.json()
    assert 1.0 <= body["predicted_rating"] <= 5.0
    assert len(body["explanation"]) == 6

    shap_magnitudes = [abs(c["shap_value"]) for c in body["explanation"]]
    assert shap_magnitudes == sorted(shap_magnitudes, reverse=True)

    # An all-zero explanation means the explainer was given a degenerate background
    # (e.g. the row being explained), which silently produces a useless explanation.
    assert any(magnitude > 0 for magnitude in shap_magnitudes)


def test_shap_values_reconcile_with_the_prediction(client, known_user_and_item):
    """SHAP is additive: base value + sum(contributions) must equal the prediction.
    If it doesn't, the explanation isn't describing the model that served the number."""
    from src.models.rating_predictor import build_explainer, load

    user_id, parent_asin = known_user_and_item
    body = client.post("/predict_rating", json={
        "user_id": user_id,
        "parent_asin": parent_asin,
        "review_text": "works really nicely, very pleased",
    }).json()

    base_value = float(np.ravel(build_explainer(load()).expected_value)[0])
    total = base_value + sum(c["shap_value"] for c in body["explanation"])

    assert total == pytest.approx(body["predicted_rating"], abs=0.01)


def test_predict_rating_handles_unknown_user_and_item(client):
    response = client.post("/predict_rating", json={
        "user_id": "UNKNOWN_USER",
        "parent_asin": "UNKNOWN_ITEM",
        "review_text": "pretty good",
    })

    assert response.status_code == 200
    assert 1.0 <= response.json()["predicted_rating"] <= 5.0
