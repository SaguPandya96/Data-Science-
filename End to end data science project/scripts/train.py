"""Train, calibrate, evaluate, and serialize the delivery-risk system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_pinball_loss,
    median_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from supplylens.calibration import ProbabilityCalibrator, expected_calibration_error
from supplylens.config import load_config, resolve_path
from supplylens.evaluation import (
    calibration_bins,
    capacity_table,
    ranking_metrics_at_capacity,
    segment_metrics,
    top_capacity_threshold,
)
from supplylens.explainability import (
    local_perturbation_contributors,
    permutation_feature_importance,
)
from supplylens.features import prepare_model_frame
from supplylens.modeling import (
    ModelBundle,
    SupplierRateBaseline,
    build_classifier,
    build_preprocessor,
    expanding_window_splits,
    save_bundle,
    temporal_split,
)
from supplylens.monitoring import build_monitoring_baseline
from supplylens.replenishment import replenishment_indicators
from supplylens.scoring import SCORING_REQUIRED_COLUMNS, score_frame

TARGET = "severe_delay"


def probability_summary(name: str, labels: pd.Series, scores: np.ndarray) -> dict:
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration_error": expected_calibration_error(labels.to_numpy(), scores),
    }


def split_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "start": frame["prediction_date"].min().date().isoformat(),
        "end": frame["prediction_date"].max().date().isoformat(),
        "positive_shipments": int(frame[TARGET].sum()),
        "prevalence": float(frame[TARGET].mean()),
    }


def cost_sensitive_policy(
    labels: np.ndarray,
    scores: np.ndarray,
    assumptions: dict,
) -> dict:
    review_cost = float(assumptions["review_cost_usd"])
    miss_exposure = float(assumptions["missed_severe_delay_exposure_usd"])
    success_rate = float(assumptions["intervention_success_rate"])
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    rows = []
    for threshold in candidates:
        flags = scores >= threshold
        true_positives = int((flags & (labels == 1)).sum())
        reviewed = int(flags.sum())
        expected_avoided = true_positives * success_rate * miss_exposure
        net_value = expected_avoided - reviewed * review_cost
        rows.append((net_value, threshold, reviewed, true_positives, expected_avoided))
    best = max(rows, key=lambda item: item[0])
    return {
        "policy": "cost_sensitive_assumption_scenario",
        "threshold": float(best[1]),
        "shipments_reviewed": int(best[2]),
        "severe_delays_captured": int(best[3]),
        "expected_avoided_exposure_usd": float(best[4]),
        "review_cost_usd": float(best[2] * review_cost),
        "expected_net_value_usd": float(best[0]),
        "assumption_based": True,
    }


def evaluate_fixed_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flags = scores >= threshold
    positives = int(labels.sum())
    captured = int((flags & (labels == 1)).sum())
    reviewed = int(flags.sum())
    prevalence = float(labels.mean())
    precision = captured / reviewed if reviewed else 0.0
    return {
        "shipments_reviewed": reviewed,
        "severe_delays_captured": captured,
        "precision": float(precision),
        "recall": float(captured / positives) if positives else 0.0,
        "lift": float(precision / prevalence) if prevalence else 0.0,
        "false_negatives": int(positives - captured),
    }


def run_lead_time_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    def eligible(frame: pd.DataFrame) -> pd.DataFrame:
        mask = (
            frame["po_sent_date"].notna()
            & frame["actual_lead_time_days"].between(0, 730)
            & frame["scheduled_lead_time_days"].between(0, 730)
        )
        return frame.loc[mask].copy()

    train_lt, validation_lt, test_lt = map(eligible, (train, validation, test))
    if min(len(train_lt), len(validation_lt), len(test_lt)) < 50:
        return (
            {
                "status": "not_run",
                "reason": "Fewer than 50 valid direct-drop observations in a temporal partition",
            },
            pd.DataFrame(),
        )

    x_train = prepare_model_frame(train_lt)
    x_validation = prepare_model_frame(validation_lt)
    x_test = prepare_model_frame(test_lt)
    y_train = train_lt["actual_lead_time_days"].to_numpy(float)
    y_validation = validation_lt["actual_lead_time_days"].to_numpy(float)
    y_test = test_lt["actual_lead_time_days"].to_numpy(float)
    preprocessor = build_preprocessor(
        int(config["modeling"].get("categorical_min_frequency", 10)), dense=True
    )

    def regressor(quantile: float) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=quantile,
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=0.5,
                        random_state=seed,
                    ),
                ),
            ]
        )

    p50_model = regressor(0.5).fit(x_train, y_train)
    p90_model = regressor(0.9).fit(x_train, y_train)
    validation_p50 = p50_model.predict(x_validation)
    validation_scheduled = validation_lt["scheduled_lead_time_days"].to_numpy(float)
    selected_p50 = "learned_p50" if mean_absolute_error(y_validation, validation_p50) < mean_absolute_error(y_validation, validation_scheduled) else "scheduled_lead_time"

    supplier_medians = train_lt.groupby("supplier")["actual_lead_time_days"].median()
    lane_key_train = train_lt["manufacturing_site"].astype(str) + " -> " + train_lt["country"].astype(str)
    lane_medians = train_lt.assign(_lane=lane_key_train).groupby("_lane")["actual_lead_time_days"].median()
    global_median = float(np.median(y_train))
    lane_key_test = test_lt["manufacturing_site"].astype(str) + " -> " + test_lt["country"].astype(str)
    predictions = {
        "scheduled_lead_time": test_lt["scheduled_lead_time_days"].to_numpy(float),
        "overall_historical_median": np.full(len(test_lt), global_median),
        "supplier_historical_median": test_lt["supplier"].map(supplier_medians).fillna(global_median).to_numpy(float),
        "lane_historical_median": lane_key_test.map(lane_medians).fillna(global_median).to_numpy(float),
        "learned_p50": p50_model.predict(x_test),
    }
    p90 = p90_model.predict(x_test)
    rows = []
    for name, predicted in predictions.items():
        rows.append(
            {
                "model": name,
                "mae_days": float(mean_absolute_error(y_test, predicted)),
                "median_absolute_error_days": float(median_absolute_error(y_test, predicted)),
                "p50_pinball_loss": float(mean_pinball_loss(y_test, predicted, alpha=0.5)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("mae_days")
    learned_beats_scheduled = float(metrics.set_index("model").loc["learned_p50", "mae_days"]) < float(metrics.set_index("model").loc["scheduled_lead_time", "mae_days"])
    recommendation = "learned_p50_candidate" if learned_beats_scheduled and selected_p50 == "learned_p50" else "retain_scheduled_lead_time_baseline"
    result = {
        "status": "completed",
        "eligible_counts": {
            "train": int(len(train_lt)),
            "validation": int(len(validation_lt)),
            "test": int(len(test_lt)),
        },
        "validation_selection": selected_p50,
        "test_metrics": metrics.to_dict(orient="records"),
        "p90_pinball_loss": float(mean_pinball_loss(y_test, p90, alpha=0.9)),
        "p90_empirical_coverage": float(np.mean(y_test <= p90)),
        "median_prediction_interval_width_days": float(np.median(np.maximum(p90 - predictions["learned_p50"], 0))),
        "recommendation": recommendation,
    }
    return result, metrics


def main() -> int:
    config = load_config()
    seed = int(config["project"]["random_seed"])
    np.random.seed(seed)
    processed_path = resolve_path(config["data"]["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError("Processed data missing. Run `python scripts/validate_data.py` first.")
    frame = pd.read_csv(
        processed_path,
        parse_dates=[
            "scheduled_delivery_date",
            "actual_delivery_date",
            "delivery_recorded_date",
            "po_sent_date",
            "prediction_date",
        ],
    )
    train, validation, test = temporal_split(frame, config["splits"])
    x_train = prepare_model_frame(train)
    x_validation = prepare_model_frame(validation)
    x_test = prepare_model_frame(test)
    y_train = train[TARGET]
    y_validation = validation[TARGET]
    y_test = test[TARGET]

    comparison_rows = []
    prevalence_scores = np.full(len(validation), y_train.mean())
    comparison_rows.append(probability_summary("prevalence_baseline", y_validation, prevalence_scores))
    supplier_baseline = SupplierRateBaseline(
        minimum_volume=int(config["operations"]["supplier_min_volume"])
    ).fit(train)
    comparison_rows.append(
        probability_summary(
            "supplier_rate_rule", y_validation, supplier_baseline.predict_proba(validation)
        )
    )

    fitted_models = {}
    validation_scores = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        model = build_classifier(name, config["modeling"], seed)
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_validation)[:, 1]
        fitted_models[name] = model
        validation_scores[name] = scores
        comparison_rows.append(probability_summary(name, y_validation, scores))
        print(f"Fitted {name}: validation PR-AUC {comparison_rows[-1]['pr_auc']:.4f}")

    comparison = pd.DataFrame(comparison_rows)
    logistic_pr = float(comparison.set_index("model").loc["logistic_regression", "pr_auc"])
    advanced_pr = float(comparison.set_index("model").loc["hist_gradient_boosting", "pr_auc"])
    selected_name = (
        "hist_gradient_boosting"
        if advanced_pr >= logistic_pr + 0.01
        else "logistic_regression"
    )
    selection_reason = (
        "Advanced model exceeded logistic-regression validation PR-AUC by at least 0.01."
        if selected_name == "hist_gradient_boosting"
        else "Advanced model did not clear the predeclared 0.01 PR-AUC improvement margin; the simpler logistic regression was retained."
    )
    selected_model = fitted_models[selected_name]
    raw_validation_scores = validation_scores[selected_name]

    calibration_rows = []
    calibrators = {}
    for method in config["modeling"]["calibration_methods"]:
        calibrator = ProbabilityCalibrator(str(method)).fit(
            raw_validation_scores, y_validation.to_numpy()
        )
        calibrated = calibrator.transform(raw_validation_scores)
        calibrators[method] = calibrator
        calibration_rows.append(
            {
                "method": method,
                "brier_score": float(brier_score_loss(y_validation, calibrated)),
                "calibration_error": expected_calibration_error(
                    y_validation.to_numpy(), calibrated
                ),
                "pr_auc": float(average_precision_score(y_validation, calibrated)),
            }
        )
    calibration_comparison = pd.DataFrame(calibration_rows).sort_values(
        ["brier_score", "calibration_error"]
    )
    selected_calibration = str(calibration_comparison.iloc[0]["method"])
    calibrator = calibrators[selected_calibration]
    validation_probabilities = calibrator.transform(raw_validation_scores)

    review_capacity = float(config["operations"]["recommended_review_capacity"])
    validation_threshold = top_capacity_threshold(validation_probabilities, review_capacity)
    bundle = ModelBundle(
        model=selected_model,
        calibrator=calibrator,
        model_name=selected_name,
        review_capacity=review_capacity,
        metadata={
            "prediction_time": "scheduled-delivery commitment",
            "target_threshold_days": int(config["target"]["delay_days_threshold"]),
            "validation_score_threshold_at_selected_capacity": validation_threshold,
        },
    )
    model_path = save_bundle(bundle, resolve_path(config["outputs"]["model_path"]))

    raw_test_probabilities = selected_model.predict_proba(x_test)[:, 1]
    test_probabilities = calibrator.transform(raw_test_probabilities)
    test_threshold = top_capacity_threshold(test_probabilities, review_capacity)
    test_metrics = ranking_metrics_at_capacity(
        y_test, test_probabilities, review_capacity
    )
    test_metrics["policy"] = f"top_{int(review_capacity * 100)}_percent_risk"

    capacities = [float(value) for value in config["operations"]["review_capacity_levels"]]
    validation_capacity = capacity_table(
        y_validation,
        validation_probabilities,
        capacities,
        validation["total_value_usd"],
    )
    test_capacity = capacity_table(
        y_test, test_probabilities, capacities, test["total_value_usd"]
    )
    assumptions = config["operations"]["scenario_assumptions"]
    cost_policy = cost_sensitive_policy(
        y_validation.to_numpy(int), validation_probabilities, assumptions
    )
    cost_policy_test = evaluate_fixed_threshold(
        y_test.to_numpy(int), test_probabilities, float(cost_policy["threshold"])
    )

    tables_dir = resolve_path(config["outputs"]["tables_dir"])
    metrics_dir = resolve_path("reports/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    calibration_comparison.to_csv(tables_dir / "calibration_comparison.csv", index=False)
    calibration_bins(y_validation, validation_probabilities).to_csv(
        tables_dir / "calibration_bins_validation.csv", index=False
    )
    calibration_bins(y_test, test_probabilities).to_csv(
        tables_dir / "calibration_bins_test.csv", index=False
    )
    validation_capacity.to_csv(tables_dir / "capacity_analysis_validation.csv", index=False)
    test_capacity.to_csv(tables_dir / "capacity_analysis_test.csv", index=False)

    scoring_input = test[SCORING_REQUIRED_COLUMNS].copy()
    scored = score_frame(bundle, scoring_input)
    evaluated = scored.merge(
        test[
            [
                "shipment_id",
                "product",
                "actual_delivery_date",
                "delivery_delay_days",
                "severe_delay",
            ]
        ],
        on="shipment_id",
        how="left",
        validate="one_to_one",
    )
    evaluated.to_csv(tables_dir / "test_scored_shipments.csv", index=False)

    queue_columns = [
        "shipment_id",
        "supplier",
        "product",
        "country",
        "shipment_mode",
        "scheduled_delivery_date",
        "predicted_severe_delay_probability",
        "risk_rank",
        "review_flag",
        "total_value_usd",
    ]
    queue = evaluated[queue_columns].copy()
    queue["top_prediction_contributors"] = "Not computed"
    top_explain_count = min(25, int(queue["review_flag"].sum()))
    for index in queue.index[:top_explain_count]:
        shipment_id = queue.loc[index, "shipment_id"]
        row = test.loc[test["shipment_id"] == shipment_id]
        contributors = local_perturbation_contributors(bundle, row, train, top_n=3)
        queue.loc[index, "top_prediction_contributors"] = "; ".join(
            f"{item['feature']} ({float(item['score_contribution']):+.3f})"
            for item in contributors
        )
    queue.to_csv(tables_dir / "shipment_intervention_queue.csv", index=False)

    scorecard = (
        evaluated.groupby("supplier", dropna=False)
        .agg(
            shipment_count=("shipment_id", "nunique"),
            severe_delay_rate=("severe_delay", "mean"),
            average_delay_days=("delivery_delay_days", "mean"),
            p90_delay_days=("delivery_delay_days", lambda values: values.quantile(0.90)),
            average_predicted_risk=("predicted_severe_delay_probability", "mean"),
            high_risk_shipment_count=("review_flag", "sum"),
            total_represented_shipment_value_usd=("total_value_usd", "sum"),
        )
        .reset_index()
    )
    minimum_volume = int(config["operations"]["supplier_min_volume"])
    scorecard["minimum_volume_eligible"] = scorecard["shipment_count"] >= minimum_volume
    scorecard["performance_priority_score"] = (
        scorecard["average_predicted_risk"] * np.log1p(scorecard["shipment_count"])
    )
    scorecard.sort_values("performance_priority_score", ascending=False).to_csv(
        tables_dir / "supplier_scorecard.csv", index=False
    )
    replenishment_indicators(frame).to_csv(
        tables_dir / "replenishment_risk_indicators.csv", index=False
    )

    test_for_segments = test.copy()
    train_supplier_counts = train["supplier"].value_counts()
    test_for_segments["supplier_volume_group"] = pd.cut(
        test_for_segments["supplier"].map(train_supplier_counts).fillna(0),
        bins=[-1, 9, 49, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    test_for_segments["seen_supplier"] = np.where(
        test_for_segments["supplier"].isin(train_supplier_counts.index), "seen", "unseen"
    )
    value_edges = np.unique(
        [-np.inf, *train["total_value_usd"].quantile([0.25, 0.5, 0.75]).tolist(), np.inf]
    )
    test_for_segments["shipment_value_band"] = pd.cut(
        test_for_segments["total_value_usd"], bins=value_edges, duplicates="drop"
    ).astype(str)
    test_for_segments["calendar_period"] = test_for_segments["prediction_date"].dt.to_period("Q").astype(str)
    review_flags = (test_probabilities >= test_threshold).astype(int)
    segments = segment_metrics(
        test_for_segments,
        test_probabilities,
        review_flags,
        (
            "supplier",
            "supplier_volume_group",
            "country",
            "shipment_mode",
            "product_group",
            "fulfill_via",
            "shipment_value_band",
            "calendar_period",
            "seen_supplier",
        ),
        minimum_volume=minimum_volume,
    )
    segments.to_csv(tables_dir / "segment_performance.csv", index=False)

    errors = evaluated.copy()
    errors["predicted_class"] = (errors["predicted_severe_delay_probability"] >= test_threshold).astype(int)
    examples = []
    cases = {
        "highest_risk": errors.sort_values("predicted_severe_delay_probability", ascending=False).head(1),
        "false_positive": errors[(errors["predicted_class"] == 1) & (errors["severe_delay"] == 0)].sort_values("predicted_severe_delay_probability", ascending=False).head(1),
        "false_negative": errors[(errors["predicted_class"] == 0) & (errors["severe_delay"] == 1)].sort_values("predicted_severe_delay_probability").head(1),
    }
    for case_type, case in cases.items():
        if not case.empty:
            record = case.iloc[0][
                [
                    "shipment_id",
                    "supplier",
                    "country",
                    "shipment_mode",
                    "predicted_severe_delay_probability",
                    "delivery_delay_days",
                    "severe_delay",
                ]
            ].to_dict()
            record["example_type"] = case_type
            examples.append(record)
    pd.DataFrame(examples).to_csv(tables_dir / "error_examples.csv", index=False)

    importance = permutation_feature_importance(bundle, test, y_test, seed)
    importance.to_csv(tables_dir / "permutation_importance.csv", index=False)

    cv_rows = []
    for train_index, validation_index, year in expanding_window_splits(train):
        fold_train = train.loc[train_index]
        fold_validation = train.loc[validation_index]
        model = build_classifier("logistic_regression", config["modeling"], seed)
        model.fit(prepare_model_frame(fold_train), fold_train[TARGET])
        scores = model.predict_proba(prepare_model_frame(fold_validation))[:, 1]
        cv_rows.append(
            {
                "validation_year": year,
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(fold_validation)),
                "validation_prevalence": float(fold_validation[TARGET].mean()),
                "pr_auc": float(average_precision_score(fold_validation[TARGET], scores)),
                "roc_auc": float(roc_auc_score(fold_validation[TARGET], scores)),
            }
        )
    pd.DataFrame(cv_rows).to_csv(tables_dir / "expanding_window_validation.csv", index=False)

    lead_time_result, lead_time_table = run_lead_time_experiment(
        train, validation, test, config, seed
    )
    lead_time_table.to_csv(tables_dir / "lead_time_model_comparison.csv", index=False)
    (metrics_dir / "lead_time_metrics.json").write_text(
        json.dumps(lead_time_result, indent=2), encoding="utf-8"
    )

    monitoring = build_monitoring_baseline(test, test_probabilities)
    resolve_path(config["outputs"]["monitoring_baseline_path"]).write_text(
        json.dumps(monitoring, indent=2), encoding="utf-8"
    )

    final_metrics = {
        "project": "SupplyLens",
        "target": {
            "threshold_days": int(config["target"]["delay_days_threshold"]),
            "definition": config["target"]["definition"],
            "modeling_grain": "ASN/DN shipment",
            "all_data_positive_shipments": int(frame[TARGET].sum()),
            "all_data_prevalence": float(frame[TARGET].mean()),
        },
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "test": split_summary(test),
        },
        "model_selection": {
            "selected_model": selected_name,
            "selection_reason": selection_reason,
            "validation_comparison": comparison.to_dict(orient="records"),
        },
        "calibration": {
            "selected_method": selected_calibration,
            "validation_comparison": calibration_comparison.to_dict(orient="records"),
        },
        "operating_policy": {
            "selected_policy": f"top_{int(review_capacity * 100)}_percent_risk_per_batch",
            "selected_capacity": review_capacity,
            "validation_score_threshold": validation_threshold,
            "validation_result": validation_capacity.loc[
                np.isclose(validation_capacity["capacity"], review_capacity)
            ].iloc[0].to_dict(),
            "test_result": test_capacity.loc[
                np.isclose(test_capacity["capacity"], review_capacity)
            ].iloc[0].to_dict(),
            "cost_sensitive_validation_scenario": cost_policy,
            "cost_sensitive_test_result_at_validation_threshold": cost_policy_test,
            "scenario_assumptions": assumptions,
        },
        "test_metrics": test_metrics,
        "lead_time_experiment": lead_time_result,
        "artifacts": {
            "model": str(Path(config["outputs"]["model_path"])),
            "intervention_queue": "reports/tables/shipment_intervention_queue.csv",
            "supplier_scorecard": "reports/tables/supplier_scorecard.csv",
            "replenishment_indicators": "reports/tables/replenishment_risk_indicators.csv",
        },
    }
    final_metrics_path = resolve_path(config["outputs"]["final_metrics_path"])
    final_metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(
        f"Selected {selected_name} with {selected_calibration} calibration. "
        f"Test PR-AUC {test_metrics['pr_auc']:.4f}; top-{review_capacity:.0%} recall "
        f"{test_metrics['recall']:.4f}."
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {final_metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise
