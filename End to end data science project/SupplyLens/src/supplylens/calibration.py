"""Validation-set probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


@dataclass
class ProbabilityCalibrator:
    method: str = "none"
    estimator: object | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        probabilities = _clip(probabilities)
        labels = np.asarray(y_true, dtype=int)
        if self.method == "none":
            self.estimator = None
        elif self.method == "sigmoid":
            logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
            model = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
            model.fit(logits, labels)
            self.estimator = model
        elif self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            model.fit(probabilities, labels)
            self.estimator = model
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = _clip(probabilities)
        if self.method == "none":
            return probabilities
        if self.estimator is None:
            raise RuntimeError("Calibrator has not been fitted")
        if self.method == "sigmoid":
            logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
            return self.estimator.predict_proba(logits)[:, 1]
        return np.asarray(self.estimator.predict(probabilities), dtype=float)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return float(error)

