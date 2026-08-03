from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#94a3b8", "#38bdf8", "#0f766e", "#f59e0b"]


def _label_bars(axis, values: list[float], suffix: str = "") -> None:
    for patch, value in zip(axis.patches, values, strict=True):
        axis.annotate(
            f"{value:,.1f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )


def save_model_comparison(performance: pd.DataFrame, destination: Path) -> Path:
    """Save a compact view of holdout error for portfolio readers."""
    labels = performance["Model"].str.replace(" ", "\n", n=1).tolist()
    rmse = performance["RMSE"].tolist()
    wape = (performance["WAPE"] * 100).tolist()
    colors = COLORS[: len(performance)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, rmse, color=colors)
    axes[0].set_title("RMSE")
    axes[0].set_ylabel("Sales / turnover units")
    _label_bars(axes[0], rmse)

    axes[1].bar(labels, wape, color=colors)
    axes[1].set_title("WAPE")
    axes[1].set_ylabel("Percent")
    _label_bars(axes[1], wape, "%")

    fig.suptitle("Six-week rolling holdout — lower is better", fontsize=14, fontweight="bold")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_scenario_impact(summary: pd.DataFrame, destination: Path) -> Path:
    """Save scenario movement relative to the model baseline."""
    changes = (summary["% Change"] * 100).tolist()
    colors = [
        "#64748b" if value == 0 else "#0f766e" if value > 0 else "#dc2626" for value in changes
    ]
    labels = summary["Scenario"].str.replace(" ", "\n", n=1).tolist()

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(labels, changes, color=colors)
    axis.axhline(0, color="#334155", linewidth=1)
    axis.set_title("Model-estimated scenario sensitivity", fontsize=14, fontweight="bold")
    axis.set_ylabel("Change from baseline (%)")
    _label_bars(axis, changes, "%")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_daily_holdout(holdout: pd.DataFrame, destination: Path) -> Path:
    """Save aggregate actual and predicted daily sales across the holdout."""
    daily = holdout.groupby("Date", as_index=False)[["Actual", "Predicted"]].sum()
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(daily["Date"], daily["Actual"], label="Actual", color="#0f172a", linewidth=2)
    axis.plot(
        daily["Date"],
        daily["Predicted"],
        label="XGBoost forecast",
        color="#0f766e",
        linewidth=2,
    )
    axis.set_title(
        "Daily portfolio forecast on the rolling holdout", fontsize=14, fontweight="bold"
    )
    axis.set_ylabel("Sales / turnover units")
    axis.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_report_figures(
    performance: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    holdout: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    return {
        "model_comparison_figure": save_model_comparison(
            performance, output_dir / "model_comparison.png"
        ),
        "scenario_impact_figure": save_scenario_impact(
            scenario_summary, output_dir / "scenario_impact.png"
        ),
        "daily_holdout_figure": save_daily_holdout(
            holdout, output_dir / "daily_holdout_forecast.png"
        ),
    }
