import pytest

from offerlift import data


def test_synthetic_frame_passes_validation(synthetic_experiment):
    data.validate(synthetic_experiment)


def test_missing_column_is_rejected(synthetic_experiment):
    with pytest.raises(ValueError, match="Missing columns"):
        data.validate(synthetic_experiment.drop(columns="visit"))


def test_conversion_without_visit_is_rejected(synthetic_experiment):
    frame = synthetic_experiment.copy()
    row = frame.index[frame["visit"] == 0][0]
    frame.loc[row, "conversion"] = 1
    with pytest.raises(ValueError, match="conversion must follow a visit"):
        data.validate(frame)


def test_contrast_keeps_two_arms(synthetic_experiment):
    subset = data.contrast(synthetic_experiment, "segment", "No E-Mail", "Mens E-Mail")
    assert set(subset["segment"]) == {"No E-Mail", "Mens E-Mail"}
    assert subset["treated"].eq(subset["segment"].eq("Mens E-Mail")).all()


def test_features_exclude_outcomes_and_assignment(synthetic_experiment):
    features = data.feature_matrix(synthetic_experiment)
    forbidden = {"visit", "conversion", "spend", "segment", "treated", "true_uplift"}
    assert forbidden.isdisjoint(features.columns)
