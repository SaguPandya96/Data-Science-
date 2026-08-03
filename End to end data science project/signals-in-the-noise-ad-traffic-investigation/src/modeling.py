from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "log_impressions",
    "log_clicks",
    "ctr",
    "conversion_per_click",
    "repeat_event_share",
    "top_source_share",
    "active_click_minute_share",
    "clicks_per_unique_user",
    "log_total_cost_units",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 3_000,
    learning_rate: float = 0.035,
    regularization: float = 0.015,
) -> np.ndarray:
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(x_with_intercept.shape[1], dtype="float64")
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    positive_weight = min(40.0, negatives / positives)
    row_weights = np.where(y == 1, positive_weight, 1.0)

    for step in range(iterations):
        probabilities = _sigmoid(x_with_intercept @ weights)
        residual = (probabilities - y) * row_weights
        gradient = (x_with_intercept.T @ residual) / row_weights.sum()
        gradient[1:] += regularization * weights[1:]
        rate = learning_rate / np.sqrt(1.0 + step / 700.0)
        weights -= rate * gradient
    return weights


def _predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([np.ones(len(x)), x]) @ weights)


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="stable")
    ordered_y = y[order]
    cumulative = np.cumsum(ordered_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * ordered_y).sum() / positives)


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = pd.Series(score).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = score >= threshold
    actual = y == 1
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    k = max(1, int(actual.sum()))
    top_k = np.argsort(-score)[:k]
    precision_at_k = float(actual[top_k].mean()) if len(top_k) else 0.0
    return {
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "average_precision": _average_precision(y, score),
        "roc_auc": _roc_auc(y, score),
        "precision_at_positive_count": precision_at_k,
    }


def _choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.10, 0.90, 81), score]))
    best = (0.0, 0.0, 0.5)
    for threshold in candidates:
        result = _classification_metrics(y, score, float(threshold))
        candidate = (result["f1"], result["precision"], float(threshold))
        if candidate[:2] > best[:2]:
            best = candidate
    return float(best[2])


def _empirical_anomaly_scores(train_clean: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict]:
    median = np.median(train_clean, axis=0)
    q25 = np.percentile(train_clean, 25, axis=0)
    q75 = np.percentile(train_clean, 75, axis=0)
    robust_scale = np.where((q75 - q25) > 1e-8, (q75 - q25) / 1.349, 1.0)

    def distance(matrix: np.ndarray) -> np.ndarray:
        z = np.abs((matrix - median) / robust_scale)
        # The two strongest deviations carry most of the signal. This is more
        # legible than a black-box detector and less diluted by quiet features.
        top_two = np.sort(z, axis=1)[:, -2:]
        return np.sqrt(np.mean(np.square(top_two), axis=1))

    reference = np.sort(distance(train_clean))
    raw = distance(values)
    percentiles = np.searchsorted(reference, raw, side="right") / max(len(reference), 1)
    artifact = {
        "median": median.tolist(),
        "robust_scale": robust_scale.tolist(),
        "reference_distance_p99": float(np.percentile(reference, 99)),
    }
    return percentiles.astype("float64"), artifact


def _add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log_impressions"] = np.log1p(result["impressions"])
    result["log_clicks"] = np.log1p(result["clicks"])
    result["log_total_cost_units"] = np.log1p(result["total_cost_units"].clip(lower=0))
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def _evidence_builder(train_clean: pd.DataFrame):
    high = train_clean[
        ["impressions", "ctr", "repeat_event_share", "top_source_share", "clicks_per_unique_user"]
    ].quantile(0.98)
    low_conversion = float(train_clean.loc[train_clean["clicks"] >= 5, "conversion_per_click"].quantile(0.10))

    def explain(row: pd.Series) -> str:
        signals: list[str] = []
        if row["impressions"] >= high["impressions"]:
            signals.append("unusually high impression volume")
        if row["ctr"] >= high["ctr"]:
            signals.append("click rate above the usual campaign-window range")
        if row["repeat_event_share"] >= high["repeat_event_share"]:
            signals.append("many events came from returning user IDs")
        if row["top_source_share"] >= high["top_source_share"]:
            signals.append("traffic was concentrated in one anonymized source group")
        if row["clicks_per_unique_user"] >= high["clicks_per_unique_user"]:
            signals.append("clicks were concentrated among relatively few users")
        if row["clicks"] >= 5 and row["conversion_per_click"] <= low_conversion:
            signals.append("clicks produced very few downstream conversions")
        if not signals:
            signals.append("several moderate deviations combined into a high score")
        return "; ".join(signals[:3])

    return explain


def score_campaign_windows(features: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _add_model_features(features)

    minimum = float(frame["window_start"].min())
    maximum = float(frame["window_start"].max())
    cutoff = minimum + float(config["train_fraction"]) * (maximum - minimum)
    train_mask = frame["window_start"] < cutoff
    test_mask = ~train_mask
    y = frame["abuse_label"].to_numpy(dtype="int64")
    train_y = y[train_mask]
    test_y = y[test_mask]
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        raise ValueError("Both train and test periods need positive and negative evaluation windows.")

    raw_x = frame[FEATURES].to_numpy(dtype="float64")
    train_x = raw_x[train_mask]
    mean = train_x.mean(axis=0)
    standard_deviation = train_x.std(axis=0)
    standard_deviation = np.where(standard_deviation > 1e-8, standard_deviation, 1.0)
    x = (raw_x - mean) / standard_deviation

    weights = _fit_logistic(x[train_mask], train_y)
    supervised_score = _predict_logistic(x, weights)

    train_clean_x = raw_x[train_mask & (y == 0)]
    anomaly_score, anomaly_artifact = _empirical_anomaly_scores(train_clean_x, raw_x)
    hybrid_score = 0.70 * supervised_score + 0.30 * anomaly_score

    supervised_threshold = _choose_threshold(train_y, supervised_score[train_mask])
    selected_threshold = _choose_threshold(train_y, hybrid_score[train_mask])
    review_threshold = float(np.clip(selected_threshold, 0.45, 0.75))
    anomaly_threshold = 0.99

    frame["supervised_score"] = supervised_score
    frame["anomaly_score"] = anomaly_score
    frame["risk_score"] = hybrid_score

    throttle_threshold = max(float(config["throttle_threshold"]), review_threshold + 0.10)
    escalate_threshold = max(float(config["escalate_threshold"]), throttle_threshold + 0.10)
    frame["recommended_action"] = np.select(
        [
            frame["risk_score"] >= escalate_threshold,
            frame["risk_score"] >= throttle_threshold,
            frame["risk_score"] >= review_threshold,
        ],
        ["escalate_for_manual_decision", "temporary_throttle_candidate", "review"],
        default="monitor",
    )

    train_clean = frame.loc[train_mask & (frame["abuse_label"] == 0)]
    explain = _evidence_builder(train_clean)
    frame["evidence_summary"] = frame.apply(explain, axis=1)
    frame["data_period"] = np.where(train_mask, "train", "test")

    metrics = {
        "data_split": {
            "cutoff_timestamp": int(cutoff),
            "train_windows": int(train_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "train_positive_windows": int(train_y.sum()),
            "test_positive_windows": int(test_y.sum()),
        },
        "anomaly_baseline_test": _classification_metrics(
            test_y, anomaly_score[test_mask], anomaly_threshold
        ),
        "supervised_test": _classification_metrics(
            test_y, supervised_score[test_mask], supervised_threshold
        ),
        "hybrid_test": _classification_metrics(test_y, hybrid_score[test_mask], review_threshold),
        "thresholds": {
            "anomaly": anomaly_threshold,
            "supervised": supervised_threshold,
            "review": review_threshold,
            "temporary_throttle_candidate": throttle_threshold,
            "escalate_for_manual_decision": escalate_threshold,
        },
    }

    test_frame = frame.loc[test_mask].copy()
    scenario_summary = (
        test_frame.groupby("evaluation_scenario", dropna=False)
        .agg(
            windows=("window_start", "size"),
            evaluation_positive_windows=("abuse_label", "sum"),
            mean_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            queued_windows=("recommended_action", lambda values: int((values != "monitor").sum())),
        )
        .reset_index()
        .sort_values("mean_risk", ascending=False)
    )

    coefficient_frame = pd.DataFrame(
        {"feature": FEATURES, "coefficient": weights[1:]}
    ).sort_values("coefficient", key=np.abs, ascending=False)

    model_artifact = {
        "model": "class-weighted logistic regression implemented with NumPy",
        "features": FEATURES,
        "feature_mean": mean.tolist(),
        "feature_standard_deviation": standard_deviation.tolist(),
        "intercept": float(weights[0]),
        "coefficients": {name: float(value) for name, value in zip(FEATURES, weights[1:])},
        "anomaly_model": anomaly_artifact,
        "hybrid_formula": "0.70 * supervised_score + 0.30 * empirical_anomaly_percentile",
        "thresholds": metrics["thresholds"],
    }

    evaluation_path = output_dir / "evaluation_predictions.csv"
    frame.to_csv(evaluation_path, index=False)
    scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    coefficient_frame.to_csv(output_dir / "model_coefficients.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "model_artifact.json").write_text(
        json.dumps(model_artifact, indent=2), encoding="utf-8"
    )

    review = frame.loc[frame["recommended_action"] != "monitor"].sort_values(
        "risk_score", ascending=False
    )
    blind_columns = [
        "window_start",
        "campaign",
        "risk_score",
        "recommended_action",
        "impressions",
        "clicks",
        "conversions",
        "ctr",
        "conversion_per_click",
        "repeat_event_share",
        "top_source_share",
        "active_click_minute_share",
        "total_cost_units",
        "evidence_summary",
        "data_period",
    ]
    review[blind_columns].to_csv(output_dir / "review_queue.csv", index=False)

    return {
        "scored": frame,
        "metrics": metrics,
        "scenario_summary": scenario_summary,
        "coefficients": coefficient_frame,
        "model_artifact": model_artifact,
        "review_queue": review,
        "cutoff": int(cutoff),
    }
