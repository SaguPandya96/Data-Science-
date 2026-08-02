"""Model definitions and chronological validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chronological_splits(
    n_rows: int,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-overlapping expanding-history train, validation, and test indices."""
    if n_rows < 3:
        raise ValueError("At least three rows are required for chronological splitting")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")

    test_start = int(n_rows * (1 - test_fraction))
    validation_start = int(test_start * (1 - validation_fraction))
    return (
        np.arange(validation_start),
        np.arange(validation_start, test_start),
        np.arange(test_start, n_rows),
    )


def candidate_models(seed: int = 42) -> dict[str, object]:
    """Return deliberately small, interpretable candidate pipelines."""
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    )
    return {"logistic_regression": logistic, "random_forest": forest}


def fit_compare(
    features: pd.DataFrame,
    target: pd.Series,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit candidates on training history and rank them on validation data only."""
    fitted: dict[str, object] = {}
    rows: list[dict[str, float | str]] = []

    for name, estimator in candidate_models().items():
        model = clone(estimator)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        probability = model.predict_proba(features.iloc[validation_idx])[:, 1]
        rows.append(
            {
                "model": name,
                "validation_roc_auc": roc_auc_score(
                    target.iloc[validation_idx], probability
                ),
                "validation_pr_auc": average_precision_score(
                    target.iloc[validation_idx], probability
                ),
            }
        )
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values(
        ["validation_roc_auc", "validation_pr_auc"], ascending=False
    )
    return fitted, comparison
