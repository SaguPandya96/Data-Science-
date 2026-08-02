"""FastAPI serving layer for the Amazon recommender + review intelligence models.

Models are loaded once at startup (they're small enough to hold in memory) and reused
across requests.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    FeatureContribution,
    PredictRatingRequest,
    PredictRatingResponse,
    RecommendationItem,
    RecommendResponse,
    SentimentRequest,
    SentimentResponse,
    SimilarItemsResponse,
)
from src.features import lookups
from src.models import rating_predictor, recommender, sentiment

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["recommender"] = recommender.load()
    state["sentiment"] = sentiment.load()
    state["rating_model"] = rating_predictor.load()
    state["lookups"] = lookups.load()
    state["shap_explainer"] = rating_predictor.build_explainer(state["rating_model"])

    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=["user_id", "parent_asin"])
    state["seen_by_user"] = train_df.groupby("user_id")["parent_asin"].apply(set).to_dict()
    yield
    state.clear()


app = FastAPI(
    title="Amazon Recommender & Review Intelligence API",
    description=(
        "Hybrid collaborative-filtering + content-based recommender trained on the "
        "Amazon Reviews 2023 (All_Beauty) dataset, with review sentiment analysis and "
        "SHAP-explained rating prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _title_for(parent_asin: str) -> str:
    return state["lookups"]["item_titles"].get(parent_asin, "Unknown product")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": sorted(state.keys())}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str, k: int = 10) -> RecommendResponse:
    model = state["recommender"]
    seen = state["seen_by_user"].get(user_id, set())
    is_cold_start = user_id not in model.cf_model.user_to_idx

    results = model.recommend(user_id, seen, k=k)
    if not is_cold_start:
        strategy = "collaborative"
    elif seen:
        strategy = "content"
    else:
        strategy = "popularity"

    return RecommendResponse(
        user_id=user_id,
        is_cold_start=is_cold_start,
        strategy=strategy,
        recommendations=[
            RecommendationItem(parent_asin=pid, product_title=_title_for(pid), score=round(score, 4))
            for pid, score in results
        ],
    )


@app.get("/similar/{parent_asin}", response_model=SimilarItemsResponse)
def similar_items(parent_asin: str, k: int = 10) -> SimilarItemsResponse:
    model = state["recommender"]
    results = model.content_model.similar_items(parent_asin, k=k)
    if not results:
        raise HTTPException(status_code=404, detail=f"Unknown product: {parent_asin}")

    return SimilarItemsResponse(
        parent_asin=parent_asin,
        product_title=_title_for(parent_asin),
        similar_items=[
            RecommendationItem(parent_asin=pid, product_title=_title_for(pid), score=round(score, 4))
            for pid, score in results
        ],
    )


@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    pipeline = state["sentiment"]
    proba = pipeline.predict_proba([request.text])[0]
    classes = list(pipeline.classes_)
    probabilities = {cls: float(p) for cls, p in zip(classes, proba, strict=True)}
    score = probabilities.get("positive", 0.0) - probabilities.get("negative", 0.0)

    return SentimentResponse(
        label=str(pipeline.predict([request.text])[0]),
        score=round(score, 4),
        probabilities={k: round(v, 4) for k, v in probabilities.items()},
    )


@app.post("/predict_rating", response_model=PredictRatingResponse)
def predict_rating(request: PredictRatingRequest) -> PredictRatingResponse:
    lookup = state["lookups"]
    sentiment_pipeline = state["sentiment"]
    model = state["rating_model"]

    global_mean = lookup["global_mean"]
    user_avg = lookup["user_avg_rating"].get(request.user_id, global_mean)
    item_avg = lookup["item_avg_rating"].get(request.parent_asin, global_mean)
    first_review = lookup["user_first_review"].get(request.user_id)
    days_since_first = (
        (lookup["max_timestamp"] - first_review).days if first_review is not None else 0
    )
    sent_score = float(
        sentiment.sentiment_score(sentiment_pipeline, [request.review_text or ""]).iloc[0]
    )

    features = pd.DataFrame([{
        "user_avg_rating": user_avg,
        "item_avg_rating": item_avg,
        "review_length": len((request.review_text or "").split()),
        "verified_purchase": int(request.verified_purchase),
        "days_since_first_review": days_since_first,
        "sentiment_score": sent_score,
    }])[rating_predictor.FEATURE_COLUMNS]

    predicted = float(model.predict(features)[0])
    shap_values = state["shap_explainer"](features)

    contributions = sorted(
        (
            FeatureContribution(
                feature=name,
                value=float(features.iloc[0][name]),
                shap_value=round(float(sv), 4),
            )
            for name, sv in zip(
                rating_predictor.FEATURE_COLUMNS, shap_values.values[0], strict=True
            )
        ),
        key=lambda c: -abs(c.shap_value),
    )

    return PredictRatingResponse(
        predicted_rating=round(predicted, 3),
        explanation=contributions,
    )
