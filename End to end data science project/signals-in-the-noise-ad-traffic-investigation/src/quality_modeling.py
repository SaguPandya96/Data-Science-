from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CTR_FEATURES = [
    "log_impressions",
    "unique_user_share",
    "repeat_event_share",
    "top_source_share",
    "log_unique_sources",
    "log_avg_cost_units",
    "hour_sin",
    "hour_cos",
    "campaign_baseline_ctr",
    "log_baseline_impressions",
    "log_history_windows",
]

CONVERSION_FEATURES = CTR_FEATURES + [
    "ctr",
    "log_clicks",
    "active_click_minute_share",
    "clicks_per_unique_user",
    "campaign_baseline_conversion_rate",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_binomial_logistic(
    x: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    iterations: int = 2_000,
    learning_rate: float = 0.12,
    regularization: float = 0.01,
) -> np.ndarray:
    """Fit an aggregated binomial model with transparent NumPy gradient descent."""
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    base_rate = float(np.clip(successes.sum() / max(trials.sum(), 1.0), 1e-5, 1 - 1e-5))
    weights = np.zeros(x_with_intercept.shape[1], dtype="float64")
    weights[0] = np.log(base_rate / (1.0 - base_rate))
    total_trials = max(float(trials.sum()), 1.0)

    for step in range(iterations):
        probability = _sigmoid(x_with_intercept @ weights)
        gradient = x_with_intercept.T @ (trials * probability - successes)
        gradient /= total_trials
        gradient[1:] += regularization * weights[1:]
        rate = learning_rate / np.sqrt(1.0 + step / 500.0)
        weights -= rate * gradient
    return weights


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    with_intercept = np.column_stack([np.ones(len(x)), x])
    return np.clip(_sigmoid(with_intercept @ weights), 1e-5, 1 - 1e-5)


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    clean_reference = np.asarray(reference, dtype="float64")
    clean_reference = clean_reference[np.isfinite(clean_reference)]
    if len(clean_reference) == 0:
        return np.zeros(len(values), dtype="float64")
    ordered = np.sort(clean_reference)
    clean_values = np.nan_to_num(values, nan=0.0, posinf=ordered[-1], neginf=ordered[0])
    return np.searchsorted(ordered, clean_values, side="right") / len(ordered)


def _weighted_metrics(successes: np.ndarray, trials: np.ndarray, probability: np.ndarray) -> dict:
    observed_rate = successes / np.maximum(trials, 1.0)
    total = max(float(trials.sum()), 1.0)
    log_loss = -np.sum(
        successes * np.log(probability) + (trials - successes) * np.log(1.0 - probability)
    ) / total
    brier = np.sum(trials * np.square(observed_rate - probability)) / total
    mae = np.sum(trials * np.abs(observed_rate - probability)) / total
    return {
        "weighted_log_loss": float(log_loss),
        "weighted_brier_score": float(brier),
        "weighted_mean_absolute_error": float(mae),
        "observed_rate": float(successes.sum() / total),
        "mean_expected_rate": float(np.sum(trials * probability) / total),
    }


def _calibration_table(
    probability: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"expected_rate": probability, "successes": successes, "trials": trials}
    )
    frame["decile"] = pd.qcut(
        frame["expected_rate"].rank(method="first"), q=min(10, len(frame)), labels=False
    )
    grouped = frame.groupby("decile", as_index=False).agg(
        windows=("expected_rate", "size"),
        expected_successes=("expected_rate", lambda values: 0.0),
        actual_successes=("successes", "sum"),
        trials=("trials", "sum"),
        min_expected_rate=("expected_rate", "min"),
        max_expected_rate=("expected_rate", "max"),
    )
    expected_by_decile = frame.assign(
        expected_successes=frame["expected_rate"] * frame["trials"]
    ).groupby("decile")["expected_successes"].sum()
    grouped["expected_successes"] = grouped["decile"].map(expected_by_decile)
    grouped["mean_expected_rate"] = grouped["expected_successes"] / grouped["trials"]
    grouped["actual_rate"] = grouped["actual_successes"] / grouped["trials"]
    grouped.insert(0, "model", model_name)
    return grouped


def _prepare_features(frame: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    train = result.loc[train_mask]
    global_ctr = float(train["clicks"].sum() / max(train["impressions"].sum(), 1))
    global_conversion_rate = float(
        train["conversions"].sum() / max(train["impressions"].sum(), 1)
    )
    median_impressions = float(max(train["impressions"].median(), 1.0))

    result["campaign_baseline_ctr"] = result["baseline_ctr"].fillna(global_ctr)
    result["campaign_baseline_conversion_rate"] = result[
        "baseline_conversion_rate"
    ].fillna(global_conversion_rate)
    result["filled_baseline_impressions"] = result["baseline_impressions"].fillna(
        median_impressions
    )
    result["log_impressions"] = np.log1p(result["impressions"].clip(lower=0))
    result["unique_user_share"] = result["unique_users"] / result["impressions"].clip(lower=1)
    result["log_unique_sources"] = np.log1p(result["unique_sources"].clip(lower=0))
    result["log_avg_cost_units"] = np.log1p(result["avg_cost_units"].clip(lower=0))
    result["hour_sin"] = np.sin(2 * np.pi * result["hour_of_day"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour_of_day"] / 24.0)
    result["log_baseline_impressions"] = np.log1p(
        result["filled_baseline_impressions"].clip(lower=0)
    )
    result["log_history_windows"] = np.log1p(result["history_windows"].clip(lower=0))
    result["log_clicks"] = np.log1p(result["clicks"].clip(lower=0))
    model_features = list(dict.fromkeys(CTR_FEATURES + CONVERSION_FEATURES))
    result[model_features] = result[model_features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    defaults = {
        "train_global_ctr": global_ctr,
        "train_global_conversion_rate": global_conversion_rate,
        "train_median_impressions": median_impressions,
    }
    return result, defaults


def _standardize(
    frame: pd.DataFrame, feature_names: list[str], train_mask: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = frame[feature_names].to_numpy(dtype="float64")
    train = raw[train_mask.to_numpy()]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (raw - mean) / scale, mean, scale


def _build_evidence(frame: pd.DataFrame, train_mask: pd.Series):
    train = frame.loc[train_mask]
    volume_high = float(train["volume_change_ratio"].dropna().quantile(0.98))
    repeat_high = float(train["repeat_event_share"].quantile(0.98))
    source_high = float(train["top_source_share"].quantile(0.98))
    cost_high = float(train["total_cost_units"].quantile(0.98))

    def explain(row: pd.Series) -> str:
        signals: list[tuple[float, str]] = []
        click_z = float(row["click_deviation_z"])
        conversion_z = float(row["conversion_deviation_z"])
        if abs(click_z) >= 2.0:
            direction = "higher" if click_z > 0 else "lower"
            signals.append((abs(click_z), f"clicks were {direction} than the expected range"))
        if conversion_z <= -2.0:
            signals.append((abs(conversion_z), "conversion-linked impressions fell below expectation"))
        if pd.notna(row["volume_change_ratio"]) and row["volume_change_ratio"] >= volume_high:
            signals.append((float(row["volume_percentile"] * 4), "volume rose sharply against the campaign's recent windows"))
        if row["repeat_event_share"] >= repeat_high:
            signals.append((float(row["concentration_percentile"] * 3), "activity was concentrated among returning user IDs"))
        if row["top_source_share"] >= source_high:
            signals.append((float(row["concentration_percentile"] * 3), "one anonymized source group dominated the window"))
        if row["total_cost_units"] >= cost_high:
            signals.append((float(row["cost_percentile"] * 2), "the window carried relatively high transformed cost exposure"))
        if row["history_windows"] < 5:
            signals.append((0.5, "the campaign had limited historical context"))
        if not signals:
            return "several moderate departures combined into an unusual quality profile"
        signals.sort(key=lambda item: item[0], reverse=True)
        return "; ".join(text for _, text in signals[:3])

    return explain


def score_observed_quality(features: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    """Model expected outcomes and rank unexplained observed-only campaign windows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = features.sort_values(["window_start", "campaign"]).reset_index(drop=True).copy()
    if len(frame) < 20:
        raise ValueError("Observed quality modeling needs at least 20 campaign windows.")

    minimum = float(frame["window_start"].min())
    maximum = float(frame["window_start"].max())
    cutoff = minimum + float(config["train_fraction"]) * (maximum - minimum)
    train_mask = frame["window_start"] < cutoff
    test_mask = ~train_mask
    if int(train_mask.sum()) < 10 or int(test_mask.sum()) < 5:
        raise ValueError("Chronological train and test periods are too small for observed quality modeling.")

    frame, defaults = _prepare_features(frame, train_mask)
    ctr_x, ctr_mean, ctr_scale = _standardize(frame, CTR_FEATURES, train_mask)
    conversion_x, conversion_mean, conversion_scale = _standardize(
        frame, CONVERSION_FEATURES, train_mask
    )
    train_index = train_mask.to_numpy()
    trials = frame["impressions"].to_numpy(dtype="float64")
    clicks = frame["clicks"].to_numpy(dtype="float64")
    conversions = frame["conversions"].to_numpy(dtype="float64")

    ctr_weights = _fit_binomial_logistic(
        ctr_x[train_index], clicks[train_index], trials[train_index]
    )
    conversion_weights = _fit_binomial_logistic(
        conversion_x[train_index], conversions[train_index], trials[train_index]
    )
    expected_ctr = _predict(ctr_x, ctr_weights)
    expected_conversion_rate = _predict(conversion_x, conversion_weights)
    expected_clicks = trials * expected_ctr
    expected_conversions = trials * expected_conversion_rate

    frame["expected_ctr"] = expected_ctr
    frame["expected_clicks"] = expected_clicks
    frame["expected_conversion_rate"] = expected_conversion_rate
    frame["expected_conversions"] = expected_conversions
    frame["click_deviation_z"] = (clicks - expected_clicks) / np.sqrt(
        trials * expected_ctr * (1.0 - expected_ctr) + 1.0
    )
    frame["conversion_deviation_z"] = (conversions - expected_conversions) / np.sqrt(
        trials * expected_conversion_rate * (1.0 - expected_conversion_rate) + 1.0
    )

    click_deviation = frame["click_deviation_z"].abs().to_numpy()
    conversion_shortfall = (-frame["conversion_deviation_z"]).clip(lower=0).to_numpy()
    volume_surge = np.log1p(frame["volume_change_ratio"].fillna(1.0).clip(lower=1.0)).to_numpy()
    concentration = np.maximum(
        frame["repeat_event_share"].to_numpy(), frame["top_source_share"].to_numpy()
    )
    cost_exposure = np.log1p(frame["total_cost_units"].clip(lower=0)).to_numpy()
    train_index_array = np.flatnonzero(train_index)

    frame["click_deviation_percentile"] = _empirical_percentile(
        click_deviation[train_index_array], click_deviation
    )
    frame["conversion_shortfall_percentile"] = _empirical_percentile(
        conversion_shortfall[train_index_array], conversion_shortfall
    )
    frame["volume_percentile"] = _empirical_percentile(
        volume_surge[train_index_array], volume_surge
    )
    frame["concentration_percentile"] = _empirical_percentile(
        concentration[train_index_array], concentration
    )
    frame["cost_percentile"] = _empirical_percentile(
        cost_exposure[train_index_array], cost_exposure
    )
    frame["raw_quality_risk"] = (
        0.35 * frame["click_deviation_percentile"]
        + 0.25 * frame["conversion_shortfall_percentile"]
        + 0.15 * frame["volume_percentile"]
        + 0.10 * frame["concentration_percentile"]
        + 0.15 * frame["cost_percentile"]
    )
    train_raw_risk = frame.loc[train_mask, "raw_quality_risk"].to_numpy()
    frame["quality_risk_score"] = _empirical_percentile(
        train_raw_risk, frame["raw_quality_risk"].to_numpy()
    )
    review_threshold = float(config.get("observed_review_quantile", 0.995))
    frame["recommended_action"] = np.where(
        frame["quality_risk_score"] >= review_threshold,
        "review_traffic_quality",
        "monitor",
    )
    frame["data_period"] = np.where(train_mask, "train", "test")
    explain = _build_evidence(frame, train_mask)
    frame["evidence_summary"] = frame.apply(explain, axis=1)

    test_index = test_mask.to_numpy()
    metrics = {
        "data_split": {
            "cutoff_timestamp": int(cutoff),
            "train_windows": int(train_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "train_campaigns": int(frame.loc[train_mask, "campaign"].nunique()),
            "test_campaigns": int(frame.loc[test_mask, "campaign"].nunique()),
        },
        "expected_click_rate_test": _weighted_metrics(
            clicks[test_index], trials[test_index], expected_ctr[test_index]
        ),
        "expected_conversion_rate_test": _weighted_metrics(
            conversions[test_index], trials[test_index], expected_conversion_rate[test_index]
        ),
        "review_queue": {
            "threshold": review_threshold,
            "eligible_test_windows": int(test_mask.sum()),
            "windows_at_or_above_threshold": int(
                ((frame["quality_risk_score"] >= review_threshold) & test_mask).sum()
            ),
            "maximum_published_windows": int(config.get("observed_review_limit", 30)),
        },
    }
    calibration = pd.concat(
        [
            _calibration_table(
                expected_ctr[test_index], clicks[test_index], trials[test_index], "click_rate"
            ),
            _calibration_table(
                expected_conversion_rate[test_index],
                conversions[test_index],
                trials[test_index],
                "conversion_rate",
            ),
        ],
        ignore_index=True,
    )
    coefficients = pd.concat(
        [
            pd.DataFrame(
                {"model": "click_rate", "feature": CTR_FEATURES, "coefficient": ctr_weights[1:]}
            ),
            pd.DataFrame(
                {
                    "model": "conversion_rate",
                    "feature": CONVERSION_FEATURES,
                    "coefficient": conversion_weights[1:],
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["model", "coefficient"], key=lambda values: values.abs() if values.name == "coefficient" else values, ascending=[True, False])

    artifact = {
        "purpose": "expected-behavior modeling for observed campaign windows",
        "training_policy": "chronological split; campaign rolling baselines contain earlier windows only",
        "click_model": {
            "model": "aggregated binomial logistic regression implemented with NumPy",
            "features": CTR_FEATURES,
            "feature_mean": ctr_mean.tolist(),
            "feature_standard_deviation": ctr_scale.tolist(),
            "intercept": float(ctr_weights[0]),
            "coefficients": {
                name: float(value) for name, value in zip(CTR_FEATURES, ctr_weights[1:])
            },
        },
        "conversion_model": {
            "model": "aggregated binomial logistic regression implemented with NumPy",
            "features": CONVERSION_FEATURES,
            "feature_mean": conversion_mean.tolist(),
            "feature_standard_deviation": conversion_scale.tolist(),
            "intercept": float(conversion_weights[0]),
            "coefficients": {
                name: float(value)
                for name, value in zip(CONVERSION_FEATURES, conversion_weights[1:])
            },
        },
        "training_defaults": defaults,
        "risk_formula": (
            "0.35 click deviation + 0.25 conversion shortfall + 0.15 volume surge + "
            "0.10 concentration + 0.15 transformed cost exposure; each component is a "
            "training-period empirical percentile"
        ),
        "review_threshold": review_threshold,
    }

    limit = int(config.get("observed_review_limit", 30))
    queue = frame.loc[
        test_mask & (frame["recommended_action"] == "review_traffic_quality")
    ].sort_values("quality_risk_score", ascending=False).head(limit)
    queue_columns = [
        "window_start",
        "campaign",
        "quality_risk_score",
        "recommended_action",
        "impressions",
        "clicks",
        "expected_clicks",
        "ctr",
        "expected_ctr",
        "conversions",
        "expected_conversions",
        "conversion_rate",
        "expected_conversion_rate",
        "click_deviation_z",
        "conversion_deviation_z",
        "volume_change_ratio",
        "repeat_event_share",
        "top_source_share",
        "total_cost_units",
        "history_windows",
        "evidence_summary",
        "data_period",
    ]

    frame.to_csv(output_dir / "observed_quality_scored_windows.csv", index=False)
    queue[queue_columns].to_csv(output_dir / "observed_review_queue.csv", index=False)
    coefficients.to_csv(output_dir / "observed_model_coefficients.csv", index=False)
    calibration.to_csv(output_dir / "observed_calibration.csv", index=False)
    (output_dir / "observed_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "observed_model_artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )

    return {
        "scored": frame,
        "review_queue": queue,
        "review_columns": queue_columns,
        "metrics": metrics,
        "model_artifact": artifact,
        "coefficients": coefficients,
        "calibration": calibration,
        "cutoff": int(cutoff),
    }
