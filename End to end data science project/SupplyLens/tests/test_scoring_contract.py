"""Production scoring input and output contract tests."""

import math

import pytest

from supplylens.calibration import ProbabilityCalibrator
from supplylens.features import prepare_model_frame
from supplylens.modeling import ModelBundle, build_classifier, temporal_split
from supplylens.scoring import SCORING_REQUIRED_COLUMNS, score_frame, validate_scoring_input


def _real_data_bundle(shipments, config):
    train, validation, _ = temporal_split(shipments, config["splits"])
    model = build_classifier("logistic_regression", config["modeling"], seed=42)
    model.fit(prepare_model_frame(train), train["severe_delay"])
    raw = model.predict_proba(prepare_model_frame(validation))[:, 1]
    calibrator = ProbabilityCalibrator("isotonic").fit(raw, validation["severe_delay"].to_numpy())
    return ModelBundle(model, calibrator, "logistic_regression", 0.20)


def test_scoring_input_rejects_missing_and_outcome_columns(shipments):
    valid = shipments.tail(20)[SCORING_REQUIRED_COLUMNS].copy()
    validate_scoring_input(valid)
    with pytest.raises(ValueError, match="missing required"):
        validate_scoring_input(valid.drop(columns="supplier"))
    prohibited = valid.assign(severe_delay=shipments.tail(20)["severe_delay"].to_numpy())
    with pytest.raises(ValueError, match="prohibited outcome"):
        validate_scoring_input(prohibited)


def test_scoring_output_contract_and_capacity(shipments, config):
    bundle = _real_data_bundle(shipments, config)
    input_frame = shipments.tail(31)[SCORING_REQUIRED_COLUMNS].copy()
    scored = score_frame(bundle, input_frame)
    assert {"predicted_severe_delay_probability", "risk_rank", "review_flag"}.issubset(scored)
    assert scored["risk_rank"].tolist() == list(range(1, 32))
    assert int(scored["review_flag"].sum()) == math.ceil(31 * 0.20)
    assert scored["predicted_severe_delay_probability"].between(0, 1).all()


def test_generated_artifacts_exist():
    from pathlib import Path

    required = [
        Path("reports/metrics/final_metrics.json"),
        Path("reports/tables/shipment_intervention_queue.csv"),
        Path("reports/tables/supplier_scorecard.csv"),
        Path("reports/figures/pr_and_roc_curves.png"),
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
