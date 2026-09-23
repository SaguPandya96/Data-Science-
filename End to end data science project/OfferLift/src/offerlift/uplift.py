"""Uplift models: estimate how much the email changes each customer's outcome."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class TLearner:
    """Fit separate outcome models for treated and control; uplift is their difference."""

    def __init__(self, base: ClassifierMixin | None = None):
        self.base = base if base is not None else default_classifier()

    def fit(self, features: pd.DataFrame, outcome: np.ndarray, treated: np.ndarray) -> TLearner:
        treated = np.asarray(treated).astype(bool)
        outcome = np.asarray(outcome)
        self.treated_model_ = clone(self.base).fit(features[treated], outcome[treated])
        self.control_model_ = clone(self.base).fit(features[~treated], outcome[~treated])
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return (
            self.treated_model_.predict_proba(features)[:, 1]
            - self.control_model_.predict_proba(features)[:, 1]
        )


class TransformedOutcome:
    """Regress Y * (T - p) / (p (1 - p)), whose expectation is the individual uplift."""

    def __init__(self, base: RegressorMixin | None = None):
        self.base = base if base is not None else default_regressor()

    def fit(
        self, features: pd.DataFrame, outcome: np.ndarray, treated: np.ndarray
    ) -> TransformedOutcome:
        treated = np.asarray(treated, dtype=float)
        propensity = treated.mean()  # constant under simple randomization
        target = np.asarray(outcome, dtype=float) * (treated - propensity) / (
            propensity * (1 - propensity)
        )
        self.model_ = clone(self.base).fit(features, target)
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(features)


def default_classifier() -> ClassifierMixin:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=0
    )


def default_regressor() -> RegressorMixin:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=0
    )


def logistic_classifier() -> ClassifierMixin:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def ridge_regressor() -> RegressorMixin:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
