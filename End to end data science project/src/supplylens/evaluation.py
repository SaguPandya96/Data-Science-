"""Metrics for imbalanced risk ranking and limited review capacity."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from supplylens.calibration import expected_calibration_error


def ranking_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float | int | list[list[int]]]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration_error": expected_calibration_error(labels, scores),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def ranking_metrics_at_capacity(
    y_true: Iterable[int], probabilities: Iterable[float], capacity: float
) -> dict[str, float | int | list[list[int]]]:
    """Evaluate an exact top-capacity policy with stable tie breaking."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    reviewed = max(1, math.ceil(len(labels) * float(capacity)))
    order = np.argsort(-scores, kind="stable")
    predictions = np.zeros(len(labels), dtype=int)
    predictions[order[:reviewed]] = 1
    threshold = float(scores[order[reviewed - 1]])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration_error": expected_calibration_error(labels, scores),
        "threshold": threshold,
        "reviewed_count": int(reviewed),
        "tie_breaking": "stable risk rank",
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def capacity_table(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    capacities: Iterable[float],
    shipment_values: Iterable[float] | None = None,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    values = (
        np.asarray(shipment_values, dtype=float)
        if shipment_values is not None
        else np.full(len(labels), np.nan)
    )
    order = np.argsort(-scores, kind="stable")
    positives = labels.sum()
    prevalence = labels.mean()
    rows = []
    for capacity in capacities:
        reviewed = max(1, math.ceil(len(labels) * float(capacity)))
        selected = order[:reviewed]
        captured = int(labels[selected].sum())
        precision = captured / reviewed
        recall = captured / positives if positives else 0.0
        rows.append(
            {
                "capacity": float(capacity),
                "shipments_reviewed": int(reviewed),
                "severe_delays_captured": captured,
                "precision": float(precision),
                "recall": float(recall),
                "lift": float(precision / prevalence) if prevalence else 0.0,
                "false_negatives": int(positives - captured),
                "represented_shipment_value_usd": float(np.nansum(values[selected])),
            }
        )
    return pd.DataFrame(rows)


def top_capacity_threshold(probabilities: Iterable[float], capacity: float) -> float:
    scores = np.asarray(probabilities, dtype=float)
    reviewed = max(1, math.ceil(len(scores) * float(capacity)))
    return float(np.sort(scores)[-reviewed])


def calibration_bins(
    y_true: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    categories = pd.cut(scores, np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        pd.DataFrame({"bin": categories, "label": labels, "score": scores})
        .groupby("bin", observed=False)
        .agg(observations=("label", "size"), observed_rate=("label", "mean"), mean_score=("score", "mean"))
        .reset_index()
    )
    table["absolute_error"] = (table["observed_rate"] - table["mean_score"]).abs()
    table["bin"] = table["bin"].astype(str)
    return table


def segment_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    review_flags: np.ndarray,
    segment_columns: tuple[str, ...],
    minimum_volume: int = 20,
) -> pd.DataFrame:
    working = frame.copy()
    working["predicted_probability"] = probabilities
    working["review_flag"] = review_flags
    outputs = []
    for column in segment_columns:
        for value, group in working.groupby(column, dropna=False):
            labels = group["severe_delay"].to_numpy(int)
            flags = group["review_flag"].to_numpy(int)
            scores = group["predicted_probability"].to_numpy(float)
            tp = int(((flags == 1) & (labels == 1)).sum())
            positives = int(labels.sum())
            reviewed = int(flags.sum())
            outputs.append(
                {
                    "segment_type": column,
                    "segment": str(value),
                    "sample_size": int(len(group)),
                    "positive_rate": float(labels.mean()),
                    "precision": float(tp / reviewed) if reviewed else np.nan,
                    "recall": float(tp / positives) if positives else np.nan,
                    "false_negative_rate": float((positives - tp) / positives) if positives else np.nan,
                    "brier_score": float(brier_score_loss(labels, scores)),
                    "reliable_volume": bool(len(group) >= minimum_volume),
                }
            )
    return pd.DataFrame(outputs)
