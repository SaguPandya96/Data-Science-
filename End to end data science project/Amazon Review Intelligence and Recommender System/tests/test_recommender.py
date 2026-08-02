import numpy as np

from src.models.recommender import ContentRecommender, HybridRecommender, MatrixFactorization


def test_matrix_factorization_learns_item_quality(sample_reviews):
    """I1 is the highest-quality item and I3 the worst; the model should rank them accordingly."""
    mf = MatrixFactorization(n_factors=8, n_epochs=60, lr=0.02, reg=0.01)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])

    scores = mf.score_items("U1", ["I1", "I2", "I3", "I4"])
    assert scores[0] > scores[2]


def test_predictions_stay_within_rating_scale(sample_reviews):
    mf = MatrixFactorization(n_factors=8, n_epochs=20)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])

    for item in ["I1", "I2", "I3", "I4"]:
        assert 1.0 <= mf.predict("U1", item) <= 5.0


def test_unknown_user_and_item_fall_back_to_global_mean(sample_reviews):
    mf = MatrixFactorization(n_factors=8, n_epochs=10)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])

    assert mf.predict("UNSEEN_USER", "UNSEEN_ITEM") == np.clip(mf.global_mean_, 1.0, 5.0)


def test_score_items_handles_unknown_items(sample_reviews):
    mf = MatrixFactorization(n_factors=8, n_epochs=10)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])

    scores = mf.score_items("U1", ["I1", "TOTALLY_NEW_ITEM"])
    assert len(scores) == 2
    assert np.isfinite(scores).all()


def test_content_recommender_returns_similar_items(sample_reviews):
    content = ContentRecommender(sample_reviews[["parent_asin", "product_title", "store"]])
    results = content.similar_items("I1", k=2)

    assert len(results) == 2
    assert all(item_id != "I1" for item_id, _ in results)


def test_content_recommender_returns_empty_for_unknown_item(sample_reviews):
    content = ContentRecommender(sample_reviews[["parent_asin", "product_title", "store"]])
    assert content.similar_items("NOT_A_REAL_ASIN") == []


def test_hybrid_excludes_already_seen_items(sample_reviews):
    mf = MatrixFactorization(n_factors=8, n_epochs=20)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])
    content = ContentRecommender(sample_reviews[["parent_asin", "product_title", "store"]])
    popularity = sample_reviews.groupby("parent_asin")["rating"].count().sort_values(ascending=False)

    hybrid = HybridRecommender(mf, content, popularity)
    recommendations = hybrid.recommend("U1", seen_items={"I1", "I2"}, k=5)

    recommended_ids = {item_id for item_id, _ in recommendations}
    assert not recommended_ids & {"I1", "I2"}


def test_hybrid_cold_start_user_still_gets_recommendations(sample_reviews):
    mf = MatrixFactorization(n_factors=8, n_epochs=20)
    mf.fit(sample_reviews["user_id"], sample_reviews["parent_asin"], sample_reviews["rating"])
    content = ContentRecommender(sample_reviews[["parent_asin", "product_title", "store"]])
    popularity = sample_reviews.groupby("parent_asin")["rating"].count().sort_values(ascending=False)

    hybrid = HybridRecommender(mf, content, popularity)
    recommendations = hybrid.recommend("BRAND_NEW_USER", seen_items=set(), k=3)

    assert len(recommendations) > 0
