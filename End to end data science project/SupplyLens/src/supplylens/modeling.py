"""Temporal splitting, baselines, model pipelines, and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from supplylens.calibration import ProbabilityCalibrator
from supplylens.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_model_frame


def temporal_split(
    frame: pd.DataFrame, split_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_column = split_config["prediction_date_column"]
    dates = pd.to_datetime(frame[date_column], errors="raise")
    train_end = pd.Timestamp(split_config["train_end"])
    validation_start = pd.Timestamp(split_config["validation_start"])
    validation_end = pd.Timestamp(split_config["validation_end"])
    test_start = pd.Timestamp(split_config["test_start"])
    test_end = pd.Timestamp(split_config["test_end"])
    train = frame.loc[dates <= train_end].copy()
    validation = frame.loc[(dates >= validation_start) & (dates <= validation_end)].copy()
    test = frame.loc[(dates >= test_start) & (dates <= test_end)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition")
    if not (
        train[date_column].max() < validation[date_column].min()
        and validation[date_column].max() < test[date_column].min()
    ):
        raise ValueError("Temporal partitions overlap or are out of order")
    return train, validation, test


def expanding_window_splits(
    frame: pd.DataFrame,
    date_column: str = "prediction_date",
    validation_years: tuple[int, ...] = (2010, 2011, 2012),
) -> Iterator[tuple[pd.Index, pd.Index, int]]:
    dates = pd.to_datetime(frame[date_column], errors="raise")
    for year in validation_years:
        train_index = frame.index[dates.dt.year < year]
        validation_index = frame.index[dates.dt.year == year]
        if len(train_index) and len(validation_index):
            yield train_index, validation_index, year


def build_preprocessor(min_frequency: int, dense: bool) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("categorical", categorical, CATEGORICAL_FEATURES), ("numeric", numeric, NUMERIC_FEATURES)],
        sparse_threshold=0.0 if dense else 0.3,
    )


def build_classifier(kind: str, config: dict[str, Any], seed: int) -> Pipeline:
    min_frequency = int(config.get("categorical_min_frequency", 10))
    if kind == "logistic_regression":
        params = config[kind]
        estimator = LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            class_weight=params.get("class_weight"),
            random_state=seed,
            solver="liblinear",
        )
        dense = False
    elif kind == "hist_gradient_boosting":
        params = config[kind]
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            class_weight=params.get("class_weight"),
            early_stopping=True,
            random_state=seed,
        )
        dense = True
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline(
        [("preprocess", build_preprocessor(min_frequency, dense=dense)), ("model", estimator)]
    )


class SupplierRateBaseline:
    """Smoothed past-period supplier-rate baseline with a global fallback."""

    def __init__(self, minimum_volume: int = 20, smoothing: float = 20.0) -> None:
        self.minimum_volume = minimum_volume
        self.smoothing = smoothing
        self.global_rate = 0.0
        self.supplier_rates: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, target: str = "severe_delay") -> "SupplierRateBaseline":
        self.global_rate = float(frame[target].mean())
        stats = frame.groupby("supplier")[target].agg(["sum", "count"])
        eligible = stats[stats["count"] >= self.minimum_volume]
        rates = (eligible["sum"] + self.smoothing * self.global_rate) / (
            eligible["count"] + self.smoothing
        )
        self.supplier_rates = rates.to_dict()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            frame["supplier"].map(self.supplier_rates).fillna(self.global_rate).to_numpy(float)
        )


@dataclass
class ModelBundle:
    model: Pipeline
    calibrator: ProbabilityCalibrator
    model_name: str
    review_capacity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = prepare_model_frame(frame)
        return self.model.predict_proba(features)[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw_proba(frame))


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_bundle(path: str | Path) -> ModelBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Saved model not found: {source}. Run `python scripts/train.py`.")
    bundle = joblib.load(source)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Saved artifact is not a SupplyLens ModelBundle")
    return bundle

