from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import load_config, resolve_path
from .data import load_inputs, merge_inputs
from .evaluation import chronological_split, segment_error
from .features import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    build_features,
    feature_columns,
    lag_column,
    numeric_feature_columns,
)
from .modeling import feature_importance, fit_candidate_models
from .monitoring import build_monitoring_baseline
from .scenarios import run_scenarios


def _prepare_output_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    paths = {
        name: resolve_path(project_root, value) for name, value in config["paths"].items()
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(
    config_path: str | Path = "config.yaml", force_download: bool = False
) -> dict[str, Path]:
    """Run ingestion through artifact persistence and return generated artifact paths."""
    config, project_root = load_config(config_path)
    output_paths = _prepare_output_paths(config, project_root)

    sales, stores, train_path, store_path = load_inputs(
        config["data"], project_root, force_download=force_download
    )
    merged = merge_inputs(sales, stores)
    lag_windows = [int(value) for value in config["features"]["lag_windows"]]
    featured = build_features(merged, lag_windows)

    train, test, cutoff = chronological_split(
        featured, int(config["validation"]["holdout_weeks"])
    )
    numeric = numeric_feature_columns(lag_windows)
    categorical = list(CATEGORICAL_FEATURES)
    columns = feature_columns(lag_windows)

    x_train = train[columns].copy()
    y_train = train[TARGET_COLUMN].copy()
    x_test = test[columns].copy()
    y_test = test[TARGET_COLUMN].copy()

    model, predictions, performance = fit_candidate_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        numeric_features=numeric,
        categorical_features=categorical,
        model_config=config["model"],
        seed=int(config.get("seed", 42)),
        naive_column=lag_column(min(lag_windows)),
    )

    model_path = output_paths["models"] / "forecasting_pipeline.joblib"
    joblib.dump(model, model_path)

    performance_path = output_paths["metrics"] / "model_performance.csv"
    performance.to_csv(performance_path, index=False)

    scenario_summary, scenario_daily = run_scenarios(
        model,
        x_test,
        test[["Date", "Store"]],
        config["scenarios"],
    )
    scenario_summary_path = output_paths["metrics"] / "scenario_summary.csv"
    scenario_daily_path = output_paths["tables"] / "scenario_daily.csv"
    scenario_summary.to_csv(scenario_summary_path, index=False)
    scenario_daily.to_csv(scenario_daily_path, index=False)

    holdout = test[["Date", "Store", TARGET_COLUMN]].reset_index(drop=True).copy()
    holdout = holdout.rename(columns={TARGET_COLUMN: "Actual"})
    holdout["Predicted"] = predictions
    holdout_path = output_paths["processed"] / "holdout_predictions.csv"
    holdout.to_csv(holdout_path, index=False)

    store_error_path = output_paths["tables"] / "store_error.csv"
    segment_error(test, y_test, predictions, "Store").to_csv(store_error_path, index=False)
    promo_error_path = output_paths["tables"] / "promotion_error.csv"
    segment_error(test, y_test, predictions, "Promo").to_csv(promo_error_path, index=False)
    weekday_error_path = output_paths["tables"] / "day_of_week_error.csv"
    segment_error(test, y_test, predictions, "DayOfWeek").to_csv(
        weekday_error_path, index=False
    )

    importance_path = output_paths["tables"] / "feature_importance.csv"
    feature_importance(model).to_csv(importance_path, index=False)

    monitoring = build_monitoring_baseline(x_train, numeric, categorical)
    monitoring["training_start"] = train["Date"].min().date().isoformat()
    monitoring["training_end"] = train["Date"].max().date().isoformat()
    monitoring_path = output_paths["metrics"] / "monitoring_baseline.json"
    _write_json(monitoring, monitoring_path)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "source_train": str(train_path),
        "source_store": str(store_path),
        "source_rows": int(len(featured)),
        "store_count": int(featured["Store"].nunique()),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "training_start": train["Date"].min().date().isoformat(),
        "training_end": train["Date"].max().date().isoformat(),
        "holdout_start": test["Date"].min().date().isoformat(),
        "holdout_end": test["Date"].max().date().isoformat(),
        "cutoff_date": cutoff.date().isoformat(),
        "model_kind": config["model"]["kind"],
        "forecast_contract": "rolling one-day-ahead forecast refreshed after actuals arrive",
        "target_semantics": "Rossmann Sales field used as a daily turnover/revenue proxy",
    }
    metadata_path = output_paths["metrics"] / "run_metadata.json"
    _write_json(metadata, metadata_path)

    return {
        "model": model_path,
        "model_performance": performance_path,
        "scenario_summary": scenario_summary_path,
        "scenario_daily": scenario_daily_path,
        "holdout_predictions": holdout_path,
        "store_error": store_error_path,
        "monitoring_baseline": monitoring_path,
        "run_metadata": metadata_path,
    }
