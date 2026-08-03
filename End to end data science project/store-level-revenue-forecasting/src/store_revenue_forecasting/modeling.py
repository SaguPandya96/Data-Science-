from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evaluation import nonnegative_predictions, regression_metrics


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_features), ("cat", categorical, categorical_features)]
    )


def build_production_estimator(model_config: dict[str, Any], seed: int) -> Any:
    kind = model_config["kind"]
    params = dict(model_config.get("params", {}))
    params.setdefault("random_state", seed)

    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    if kind == "random_forest":
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported model kind: {kind}")


def build_pipeline(
    estimator: Any, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def fit_candidate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    model_config: dict[str, Any],
    seed: int,
    naive_column: str,
) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    """Compare a planning rule, linear baseline, and configured production model."""
    rows: list[dict[str, float | str]] = []

    naive_predictions = nonnegative_predictions(x_test[naive_column])
    rows.append(
        {"Model": "Naive recent-demand average", **regression_metrics(y_test, naive_predictions)}
    )

    linear = build_pipeline(LinearRegression(), numeric_features, categorical_features)
    linear.fit(x_train, y_train)
    linear_predictions = nonnegative_predictions(linear.predict(x_test))
    rows.append({"Model": "Linear Regression", **regression_metrics(y_test, linear_predictions)})

    production = build_pipeline(
        build_production_estimator(model_config, seed), numeric_features, categorical_features
    )
    production.fit(x_train, y_train)
    production_predictions = nonnegative_predictions(production.predict(x_test))
    rows.append(
        {
            "Model": str(model_config.get("name", model_config["kind"])),
            **regression_metrics(y_test, production_predictions),
        }
    )

    return production, production_predictions, pd.DataFrame(rows)


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return transformed feature importance when the estimator exposes it."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    names = model.named_steps["preprocess"].get_feature_names_out()
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in names]
    values = np.asarray(estimator.feature_importances_, dtype=float)
    if len(clean_names) != len(values):
        raise ValueError("Feature names and estimator importance values have different lengths.")
    return (
        pd.DataFrame({"Feature": clean_names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
