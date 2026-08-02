from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    parent_asin: str
    product_title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: str
    is_cold_start: bool = Field(description="True if the user was unseen during training")
    strategy: str = Field(
        description="Which path produced these: 'collaborative', 'content', or 'popularity'"
    )
    recommendations: list[RecommendationItem]


class SimilarItemsResponse(BaseModel):
    parent_asin: str
    product_title: str
    similar_items: list[RecommendationItem]


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, description="Review text to analyze")


class SentimentResponse(BaseModel):
    label: str
    score: float = Field(description="P(positive) - P(negative), in [-1, 1]")
    probabilities: dict[str, float]


class PredictRatingRequest(BaseModel):
    user_id: str
    parent_asin: str
    review_text: str = ""
    verified_purchase: bool = True


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class PredictRatingResponse(BaseModel):
    predicted_rating: float
    explanation: list[FeatureContribution] = Field(
        description="SHAP contributions, largest absolute impact first"
    )
