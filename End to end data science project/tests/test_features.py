"""Reusable feature-engineering tests."""

import numpy as np

from supplylens.features import MODEL_FEATURES, prepare_model_frame


def test_preprocessing_creates_exact_feature_allowlist(shipments):
    features = prepare_model_frame(shipments.head(100))
    assert features.columns.tolist() == MODEL_FEATURES
    assert len(features) == 100


def test_log_features_are_finite_when_observed(shipments):
    features = prepare_model_frame(shipments.head(500))
    log_columns = [column for column in features if column.startswith("log_")]
    for column in log_columns:
        observed = features[column].dropna().to_numpy(float)
        assert np.isfinite(observed).all()
        assert (observed >= 0).all()


def test_unknown_category_transformation_is_supported(shipments, config):
    from supplylens.modeling import build_classifier

    model = build_classifier("logistic_regression", config["modeling"], seed=42)
    train_features = prepare_model_frame(shipments.head(600))
    model.fit(train_features, shipments.head(600)["severe_delay"])
    unseen = shipments.iloc[[700]].copy()
    unseen["supplier"] = "Previously Unobserved Supplier"
    probability = model.predict_proba(prepare_model_frame(unseen))[0, 1]
    assert 0 <= probability <= 1
