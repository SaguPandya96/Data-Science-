"""Non-causal global and local prediction explanations."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from supplylens.features import MODEL_FEATURES, prepare_model_frame
from supplylens.modeling import ModelBundle


def permutation_feature_importance(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    y_true: pd.Series,
    seed: int,
    repeats: int = 8,
) -> pd.DataFrame:
    features = prepare_model_frame(frame)
    result = permutation_importance(
        bundle.model,
        features,
        y_true,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def local_perturbation_contributors(
    bundle: ModelBundle,
    row: pd.DataFrame,
    reference_frame: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    """Measure local score changes when each input is replaced by a reference value."""
    base_score = float(bundle.predict_proba(row)[0])
    prepared_reference = prepare_model_frame(reference_frame)
    prepared_row = prepare_model_frame(row)
    records = []
    for column in MODEL_FEATURES:
        perturbed = prepared_row.copy()
        if column in prepared_reference.select_dtypes(include=["number"]).columns:
            replacement = prepared_reference[column].median()
        else:
            mode = prepared_reference[column].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else "Unknown"
        perturbed[column] = replacement
        raw_score = bundle.model.predict_proba(perturbed)[:, 1]
        perturbed_score = float(bundle.calibrator.transform(raw_score)[0])
        records.append(
            {
                "feature": column,
                "score_contribution": base_score - perturbed_score,
            }
        )
    records.sort(key=lambda item: abs(float(item["score_contribution"])), reverse=True)
    return records[:top_n]

