"""Build EDA, evaluation, causal-readiness, and scenario artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from supplylens.config import load_config, resolve_path

os.environ.setdefault("MPLCONFIGDIR", str(resolve_path(".cache/matplotlib")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from supplylens.modeling import load_bundle, temporal_split  # noqa: E402
from supplylens.monitoring import (  # noqa: E402
    category_frequency_distance,
    population_stability_index,
)

COLORS = {
    "navy": "#12263A",
    "blue": "#2E86AB",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "gray": "#6B7280",
}


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def smd(values: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, dtype=float)

    def weighted_stats(mask: np.ndarray) -> tuple[float, float]:
        w = weights[mask]
        x = values[mask]
        mean = np.average(x, weights=w)
        variance = np.average((x - mean) ** 2, weights=w)
        return mean, variance

    mean_t, variance_t = weighted_stats(treatment == 1)
    mean_c, variance_c = weighted_stats(treatment == 0)
    pooled = np.sqrt((variance_t + variance_c) / 2)
    return float((mean_t - mean_c) / pooled) if pooled > 0 else 0.0


def causal_readiness(frame: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> dict:
    treatment = (frame["fulfill_via"] == "From RDC").astype(int).to_numpy()
    design = pd.DataFrame(
        {
            "log_total_value": np.log1p(frame["total_value_usd"].clip(lower=0)),
            "log_total_quantity": np.log1p(frame["total_quantity"].clip(lower=0)),
            "scheduled_year": frame["prediction_date"].dt.year.astype(str),
            "country": frame["country"].astype(str),
            "product_group": frame["product_group"].astype(str),
        }
    )
    design = pd.get_dummies(
        design,
        columns=["scheduled_year", "country", "product_group"],
        drop_first=False,
        dtype=float,
    )
    propensity_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.3, random_state=42)
    )
    propensity_model.fit(design, treatment)
    propensity = np.clip(propensity_model.predict_proba(design)[:, 1], 0.005, 0.995)
    auc = float(roc_auc_score(treatment, propensity))
    weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
    weights = np.minimum(weights, np.quantile(weights, 0.99))

    balance_rows = []
    for column in design.columns:
        before = smd(design[column].to_numpy(), treatment)
        after = smd(design[column].to_numpy(), treatment, weights)
        balance_rows.append(
            {
                "covariate": column,
                "absolute_smd_before": abs(before),
                "absolute_smd_after_weighting": abs(after),
            }
        )
    balance = pd.DataFrame(balance_rows).sort_values(
        "absolute_smd_before", ascending=False
    )
    balance.to_csv(tables_dir / "causal_balance_diagnostics.csv", index=False)

    outcome = frame["severe_delay"].to_numpy(float)
    trimming_rows = []
    for trim in (0.01, 0.05, 0.10):
        keep = (propensity >= trim) & (propensity <= 1 - trim)
        kept_treatment = treatment[keep]
        kept_outcome = outcome[keep]
        kept_propensity = propensity[keep]
        kept_weights = np.where(
            kept_treatment == 1, 1 / kept_propensity, 1 / (1 - kept_propensity)
        )
        if len(np.unique(kept_treatment)) < 2:
            difference = np.nan
        else:
            treated_mean = np.average(
                kept_outcome[kept_treatment == 1],
                weights=kept_weights[kept_treatment == 1],
            )
            control_mean = np.average(
                kept_outcome[kept_treatment == 0],
                weights=kept_weights[kept_treatment == 0],
            )
            difference = treated_mean - control_mean
        trimming_rows.append(
            {
                "trim_bound": trim,
                "retained_rows": int(keep.sum()),
                "retained_fraction": float(keep.mean()),
                "weighted_risk_difference_association": float(difference),
            }
        )
    trimming = pd.DataFrame(trimming_rows)
    trimming.to_csv(tables_dir / "causal_trimming_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(
        propensity[treatment == 0],
        bins=25,
        alpha=0.6,
        density=True,
        label="Direct Drop",
        color=COLORS["blue"],
    )
    ax.hist(
        propensity[treatment == 1],
        bins=25,
        alpha=0.6,
        density=True,
        label="From RDC",
        color=COLORS["orange"],
    )
    ax.set(title="Propensity-score overlap is limited", xlabel="Estimated propensity for From RDC", ylabel="Density")
    ax.legend()
    save_figure(fig, figures_dir / "causal_overlap.png")

    return {
        "treatment": "Fulfill Via equals From RDC",
        "outcome": "Severe delay greater than 7 days",
        "propensity_auc": auc,
        "below_0_05_or_above_0_95_fraction": float(
            ((propensity < 0.05) | (propensity > 0.95)).mean()
        ),
        "maximum_absolute_smd_before": float(balance["absolute_smd_before"].max()),
        "maximum_absolute_smd_after_weighting": float(
            balance["absolute_smd_after_weighting"].max()
        ),
        "trimming_sensitivity": trimming.to_dict(orient="records"),
        "identification_conclusion": (
            "No causal effect is reported. Fulfillment assignment has limited overlap and is confounded by "
            "unobserved inventory availability, urgency, contracting, and routing decisions."
        ),
    }


def business_impact_sensitivity(
    capacity: pd.DataFrame, assumptions: dict, tables_dir: Path
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for _, policy in capacity.iterrows():
        for success_rate in (0.10, 0.25, 0.50):
            for miss_exposure in (1000, 2500, 5000):
                review_cost = float(assumptions["review_cost_usd"])
                total_review_cost = float(policy["shipments_reviewed"] * review_cost)
                expected_avoided = float(
                    policy["severe_delays_captured"] * success_rate * miss_exposure
                )
                captured = float(policy["severe_delays_captured"])
                break_even = (
                    total_review_cost / (captured * miss_exposure) if captured else np.nan
                )
                rows.append(
                    {
                        "capacity": float(policy["capacity"]),
                        "intervention_success_rate_assumption": success_rate,
                        "missed_delay_exposure_usd_assumption": miss_exposure,
                        "review_cost_usd": total_review_cost,
                        "expected_avoided_exposure_usd": expected_avoided,
                        "expected_net_value_usd": expected_avoided - total_review_cost,
                        "break_even_success_rate": break_even,
                    }
                )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(tables_dir / "business_impact_sensitivity.csv", index=False)
    selected_capacity = 0.20
    selected = sensitivity[
        np.isclose(sensitivity["capacity"], selected_capacity)
        & np.isclose(
            sensitivity["intervention_success_rate_assumption"],
            float(assumptions["intervention_success_rate"]),
        )
        & np.isclose(
            sensitivity["missed_delay_exposure_usd_assumption"],
            float(assumptions["missed_severe_delay_exposure_usd"]),
        )
    ].iloc[0]
    summary = selected.to_dict()
    summary["interpretation"] = (
        "Assumption-based scenario only; these values are not observed costs or realized savings."
    )
    return summary, sensitivity


def main() -> int:
    configure_plots()
    config = load_config()
    figures_dir = resolve_path(config["outputs"]["figures_dir"])
    tables_dir = resolve_path(config["outputs"]["tables_dir"])
    metrics_dir = resolve_path("reports/metrics")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(
        resolve_path(config["data"]["processed_path"]),
        parse_dates=[
            "scheduled_delivery_date",
            "actual_delivery_date",
            "delivery_recorded_date",
            "po_sent_date",
            "prediction_date",
        ],
    )
    train, validation, test = temporal_split(frame, config["splits"])
    scored = pd.read_csv(
        tables_dir / "test_scored_shipments.csv",
        parse_dates=["scheduled_delivery_date", "actual_delivery_date"],
    )
    probabilities = scored["predicted_severe_delay_probability"].to_numpy(float)
    labels = scored["severe_delay"].to_numpy(int)

    missingness = (
        frame.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index()
    )
    missingness.columns = ["column", "missing_rate"]
    missingness.to_csv(tables_dir / "missingness_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_missing = missingness.head(12).sort_values("missing_rate")
    ax.barh(top_missing["column"], top_missing["missing_rate"], color=COLORS["blue"])
    ax.set(title="Missingness is concentrated in cost and PO-date fields", xlabel="Missing rate", ylabel="")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save_figure(fig, figures_dir / "missingness.png")

    prevalence = pd.DataFrame(
        {
            "delay_threshold_days": [0, 3, 7, 14],
            "positive_shipments": [
                int(frame[f"late_gt_{threshold}_days"].sum()) for threshold in (0, 3, 7, 14)
            ],
            "prevalence": [
                float(frame[f"late_gt_{threshold}_days"].mean()) for threshold in (0, 3, 7, 14)
            ],
        }
    )
    prevalence.to_csv(tables_dir / "target_threshold_prevalence.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(prevalence["delay_threshold_days"].astype(str), prevalence["prevalence"], color=[COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["red"]])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in prevalence["prevalence"]], padding=4)
    ax.set(title="Severe-delay prevalence by candidate threshold", xlabel="Days late (strictly greater than)", ylabel="Shipment prevalence", ylim=(0, prevalence["prevalence"].max() * 1.25))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save_figure(fig, figures_dir / "target_prevalence.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    clipped_delay = frame["delivery_delay_days"].clip(-120, 120)
    ax.hist(clipped_delay, bins=60, color=COLORS["blue"], alpha=0.85)
    ax.axvline(7, color=COLORS["red"], linestyle="--", label="Severe-delay threshold")
    ax.set(title="Delivery-delay distribution (tails clipped for visibility)", xlabel="Actual minus scheduled delivery (days)", ylabel="Shipments")
    ax.legend()
    save_figure(fig, figures_dir / "delivery_delay_distribution.png")

    annual = (
        frame.assign(year=frame["prediction_date"].dt.year)
        .groupby("year")
        .agg(shipments=("shipment_id", "count"), severe_delay_rate=("severe_delay", "mean"))
        .reset_index()
    )
    annual.to_csv(tables_dir / "target_drift_by_year.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(annual["year"], annual["shipments"], color=COLORS["blue"], alpha=0.45, label="Shipments")
    ax1.set_ylabel("Shipments")
    ax2 = ax1.twinx()
    ax2.plot(annual["year"], annual["severe_delay_rate"], color=COLORS["red"], marker="o", linewidth=2.5, label="Severe-delay rate")
    ax2.set_ylabel("Severe-delay rate")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax1.set(title="Shipment volume and target rate vary substantially over time", xlabel="Scheduled-delivery year")
    save_figure(fig, figures_dir / "volume_and_target_drift.png")

    supplier = (
        frame.groupby("supplier")
        .agg(shipments=("shipment_id", "count"), severe_delay_rate=("severe_delay", "mean"), average_delay_days=("delivery_delay_days", "mean"))
        .reset_index()
    )
    supplier["minimum_volume_eligible"] = supplier["shipments"] >= int(
        config["operations"]["supplier_min_volume"]
    )
    supplier.to_csv(tables_dir / "supplier_performance_history.csv", index=False)
    eligible = supplier[supplier["minimum_volume_eligible"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(eligible["shipments"], eligible["severe_delay_rate"], s=np.sqrt(eligible["shipments"]) * 12, alpha=0.7, color=COLORS["teal"])
    ax.axhline(frame["severe_delay"].mean(), color=COLORS["red"], linestyle="--", label="Portfolio rate")
    ax.set(title="Eligible suppliers differ in observed delay rate", xlabel="Historical shipments", ylabel="Severe-delay rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    save_figure(fig, figures_dir / "supplier_performance.png")

    for column, filename in (
        ("country", "destination_performance.csv"),
        ("shipment_mode", "shipment_mode_performance.csv"),
        ("product_group", "product_group_performance.csv"),
        ("fulfill_via", "fulfillment_performance.csv"),
    ):
        table = (
            frame.groupby(column)
            .agg(shipments=("shipment_id", "count"), severe_delay_rate=("severe_delay", "mean"), average_delay_days=("delivery_delay_days", "mean"), represented_value_usd=("total_value_usd", "sum"))
            .reset_index()
            .sort_values("shipments", ascending=False)
        )
        table.to_csv(tables_dir / filename, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    mode = pd.read_csv(tables_dir / "shipment_mode_performance.csv")
    bars = ax.bar(mode["shipment_mode"], mode["severe_delay_rate"], color=COLORS["orange"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in mode["severe_delay_rate"]], padding=3, fontsize=9)
    ax.set(title="Observed severe-delay rate by shipment mode", xlabel="Shipment mode", ylabel="Severe-delay rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save_figure(fig, figures_dir / "shipment_mode_performance.png")

    fig, ax = plt.subplots(figsize=(9, 2.7))
    split_rows = [("Train", train, COLORS["blue"]), ("Validation", validation, COLORS["orange"]), ("Test", test, COLORS["teal"])]
    for index, (name, split, color) in enumerate(split_rows):
        start, end = split["prediction_date"].min(), split["prediction_date"].max()
        ax.barh(index, (end - start).days, left=start, height=0.55, color=color)
        ax.text(start + (end - start) / 2, index, f"{name}: {len(split):,}", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
    ax.set_yticks([])
    ax.set(title="Chronological train / validation / final-test design", xlabel="Scheduled-delivery date")
    save_figure(fig, figures_dir / "temporal_split_timeline.png")

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(recall, precision, color=COLORS["blue"], linewidth=2.5)
    axes[0].axhline(labels.mean(), color=COLORS["gray"], linestyle="--", label="Test prevalence")
    axes[0].set(title=f"Precision–recall (PR-AUC {average_precision_score(labels, probabilities):.3f})", xlabel="Recall", ylabel="Precision")
    axes[0].legend()
    axes[1].plot(fpr, tpr, color=COLORS["teal"], linewidth=2.5)
    axes[1].plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--")
    axes[1].set(title=f"ROC (AUC {roc_auc_score(labels, probabilities):.3f})", xlabel="False-positive rate", ylabel="True-positive rate")
    save_figure(fig, figures_dir / "pr_and_roc_curves.png")

    observed, predicted = calibration_curve(labels, probabilities, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", linewidth=2.5, color=COLORS["blue"], label="Isotonic test bins")
    ax.set(title="Calibration on the untouched test period", xlabel="Mean predicted probability", ylabel="Observed severe-delay rate", xlim=(0, max(0.45, predicted.max() * 1.1)), ylim=(0, max(0.45, observed.max() * 1.1)))
    ax.legend()
    save_figure(fig, figures_dir / "calibration_plot.png")

    ordered = scored.sort_values("predicted_severe_delay_probability", ascending=False, kind="stable")
    ordered["population_fraction"] = np.arange(1, len(ordered) + 1) / len(ordered)
    ordered["cumulative_gain"] = ordered["severe_delay"].cumsum() / ordered["severe_delay"].sum()
    ordered["cumulative_lift"] = ordered["cumulative_gain"] / ordered["population_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(ordered["population_fraction"], ordered["cumulative_gain"], color=COLORS["blue"], linewidth=2.5)
    axes[0].plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--")
    axes[0].set(title="Cumulative gains", xlabel="Fraction reviewed", ylabel="Severe delays captured")
    axes[1].plot(ordered["population_fraction"], ordered["cumulative_lift"], color=COLORS["teal"], linewidth=2.5)
    axes[1].axhline(1, color=COLORS["gray"], linestyle="--")
    axes[1].set(title="Cumulative lift", xlabel="Fraction reviewed", ylabel="Lift", ylim=(0, max(3, ordered.loc[ordered["population_fraction"] >= 0.02, "cumulative_lift"].max() * 1.1)))
    save_figure(fig, figures_dir / "gains_and_lift.png")

    flags = scored["review_flag"].to_numpy(int)
    matrix = np.array(
        [
            [int(((flags == 0) & (labels == 0)).sum()), int(((flags == 1) & (labels == 0)).sum())],
            [int(((flags == 0) & (labels == 1)).sum()), int(((flags == 1) & (labels == 1)).sum())],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(matrix, display_labels=["Not severe", "Severe"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Top-20% review policy confusion matrix")
    save_figure(fig, figures_dir / "confusion_matrix.png")

    importance = pd.read_csv(tables_dir / "permutation_importance.csv").head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.barh(importance["feature"], importance["importance_mean"], xerr=importance["importance_std"], color=COLORS["teal"])
    ax.axvline(0, color=COLORS["gray"], linewidth=1)
    ax.set(title="Permutation importance on final-test PR-AUC", xlabel="Mean change in PR-AUC when permuted", ylabel="")
    save_figure(fig, figures_dir / "permutation_importance.png")

    lead = pd.read_csv(tables_dir / "lead_time_model_comparison.csv").sort_values("mae_days", ascending=False)
    if not lead.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.barh(lead["model"], lead["mae_days"], color=[COLORS["red"] if name == "learned_p50" else COLORS["blue"] for name in lead["model"]])
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in lead["mae_days"]], padding=3, fontsize=9)
        ax.set(title="Scheduled lead time remains the strongest test baseline", xlabel="Mean absolute error (days)", ylabel="")
        save_figure(fig, figures_dir / "lead_time_model_comparison.png")

    causal = causal_readiness(frame, tables_dir, figures_dir)
    (metrics_dir / "causal_readiness.json").write_text(
        json.dumps(causal, indent=2), encoding="utf-8"
    )

    capacity = pd.read_csv(tables_dir / "capacity_analysis_test.csv")
    impact_summary, _ = business_impact_sensitivity(
        capacity, config["operations"]["scenario_assumptions"], tables_dir
    )
    (metrics_dir / "business_impact_scenario.json").write_text(
        json.dumps(impact_summary, indent=2), encoding="utf-8"
    )

    bundle = load_bundle(resolve_path(config["outputs"]["model_path"]))
    train_probabilities = bundle.predict_proba(train)
    monitoring_example = {
        "numeric_psi_train_to_test": {
            column: population_stability_index(train[column], test[column])
            for column in ("total_quantity", "total_value_usd", "weight_kg", "freight_cost_usd")
        },
        "category_frequency_distance_train_to_test": {
            column: category_frequency_distance(train[column], test[column])
            for column in ("supplier", "country", "shipment_mode", "fulfill_via", "product_group")
        },
        "prediction_psi_train_to_test": population_stability_index(
            train_probabilities, probabilities
        ),
        "train_high_risk_rate_at_test_score_cutoff": float(
            (train_probabilities >= np.quantile(probabilities, 0.80)).mean()
        ),
        "test_review_queue_rate": float(scored["review_flag"].mean()),
    }
    (metrics_dir / "monitoring_drift_example.json").write_text(
        json.dumps(monitoring_example, indent=2), encoding="utf-8"
    )

    final_metrics_path = resolve_path(config["outputs"]["final_metrics_path"])
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    final_metrics["causal_readiness"] = causal
    final_metrics["business_impact_scenario"] = impact_summary
    final_metrics["report_artifacts"] = {
        "figure_count": len(list(figures_dir.glob("*.png"))),
        "table_count": len(list(tables_dir.glob("*.csv"))),
    }
    final_metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(
        f"Built {len(list(figures_dir.glob('*.png')))} figures and "
        f"{len(list(tables_dir.glob('*.csv')))} tables."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

