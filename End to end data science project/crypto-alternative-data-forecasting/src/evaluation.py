"""Evaluation metrics and time-series-aware uncertainty estimates."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, prediction, probability) -> dict[str, float]:
    """Calculate thresholded and ranking metrics for a binary forecast."""
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "mcc": matthews_corrcoef(y_true, prediction),
        "directional_accuracy": accuracy_score(y_true, prediction),
    }


def regression_metrics(y_true, prediction) -> dict[str, float]:
    """Calculate scale, fit, and rank metrics for a return forecast."""
    return {
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": mean_squared_error(y_true, prediction) ** 0.5,
        "r2": r2_score(y_true, prediction),
        "spearman_ic": spearmanr(y_true, prediction).statistic,
    }


def moving_block_accuracy_interval(
    y_true,
    prediction,
    block_size: int = 7,
    repetitions: int = 1_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a 95% moving-block bootstrap interval for directional accuracy."""
    truth = np.asarray(y_true)
    forecast = np.asarray(prediction)
    if truth.size != forecast.size or truth.size < block_size:
        raise ValueError("Inputs must have equal length and contain at least one block")

    rng = np.random.default_rng(seed)
    starts = np.arange(truth.size - block_size + 1)
    estimates = []
    blocks_needed = int(np.ceil(truth.size / block_size))
    for _ in range(repetitions):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block_size) for start in selected]
        )[: truth.size]
        estimates.append(accuracy_score(truth[indices], forecast[indices]))
    return tuple(np.quantile(estimates, [0.025, 0.975]))
