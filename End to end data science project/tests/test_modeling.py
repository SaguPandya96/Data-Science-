"""Model pipeline, calibration, and serialization tests."""

import numpy as np

from supplylens.calibration import ProbabilityCalibrator
from supplylens.features import prepare_model_frame
from supplylens.modeling import (
    ModelBundle,
    build_classifier,
    load_bundle,
    save_bundle,
    temporal_split,
)


def _fit_bundle(shipments, config):
    train, validation, _ = temporal_split(shipments, config["splits"])
    model = build_classifier("logistic_regression", config["modeling"], seed=42)
    model.fit(prepare_model_frame(train), train["severe_delay"])
    validation_probability = model.predict_proba(prepare_model_frame(validation))[:, 1]
    calibrator = ProbabilityCalibrator("sigmoid").fit(
        validation_probability, validation["severe_delay"].to_numpy()
    )
    return ModelBundle(model, calibrator, "test_logistic", 0.20)


def test_model_probabilities_are_bounded(shipments, config):
    bundle = _fit_bundle(shipments, config)
    probabilities = bundle.predict_proba(shipments.tail(25))
    assert len(probabilities) == 25
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_saved_bundle_round_trip(shipments, config, tmp_path):
    bundle = _fit_bundle(shipments, config)
    output = tmp_path / "bundle.joblib"
    save_bundle(bundle, output)
    loaded = load_bundle(output)
    expected = bundle.predict_proba(shipments.tail(10))
    observed = loaded.predict_proba(shipments.tail(10))
    np.testing.assert_allclose(observed, expected)
