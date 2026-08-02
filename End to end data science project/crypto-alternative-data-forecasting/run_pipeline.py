"""Rebuild the cached dataset, fitted model, figures, and evaluation reports."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.inspection import permutation_importance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtesting import backtest
from src.data_cleaning import clean_market, clean_news
from src.data_ingestion import download_gdelt_news, download_market_data
from src.evaluation import classification_metrics, moving_block_accuracy_interval
from src.features import build_market_features, merge_point_in_time
from src.models import candidate_models, chronological_splits, fit_compare
from src.sentiment import aggregate_daily, score_vader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger(__name__)
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
np.random.seed(CONFIG["seed"])

for relative_path in [
    "data/raw",
    "data/interim",
    "data/processed",
    "reports/figures",
    "models",
]:
    (ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def save_line_chart(filename: str, title: str, x, y, ylabel: str) -> None:
    """Save one consistently formatted line chart."""
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x, y)
    axis.set(title=title, xlabel="Date", ylabel=ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ROOT / f"reports/figures/{filename}.png", dpi=140)
    plt.close(figure)


raw_market = download_market_data(
    CONFIG["data"]["ticker"],
    CONFIG["data"]["start_date"],
    CONFIG["data"]["end_date"],
    ROOT / "data/raw/bitcoin_market_data.csv",
)

# GDELT DOC is a rolling recent-news service rather than a full archive.
news_end = pd.Timestamp(CONFIG["data"]["end_date"])
news_start = (news_end - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
raw_news = download_gdelt_news(
    CONFIG["data"]["news_query"],
    news_start,
    CONFIG["data"]["end_date"],
    ROOT / "data/raw/crypto_news.csv",
    chunk_days=15,
    max_records=250,
)

market = clean_market(raw_market)
market.to_csv(ROOT / "data/interim/cleaned_market_data.csv", index=False)

news = clean_news(raw_news, CONFIG["data"]["news_cutoff_utc_hour"])
scored_news = score_vader(news)
scored_news.to_csv(ROOT / "data/interim/scored_news_data.csv", index=False)

daily_news = aggregate_daily(scored_news)
market_with_features = build_market_features(market)
dataset = merge_point_in_time(market_with_features, daily_news)
dataset = dataset.dropna(subset=["target"]).reset_index(drop=True)
dataset["target"] = dataset["target"].astype(int)
dataset.to_csv(ROOT / "data/processed/modelling_dataset.csv", index=False)

market_features = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_volatility_7",
    "rolling_volatility_14",
    "rolling_volatility_30",
    "volume_change",
    "volume_zscore_30",
    "momentum_7",
    "momentum_30",
    "rsi_14",
    "macd",
    "bollinger_position",
    "ma_ratio_7",
    "ma_ratio_14",
    "ma_ratio_30",
    "high_low_range",
    "open_close_spread",
    "rolling_skew_30",
    "rolling_kurtosis_30",
    "drawdown",
    "volatility_regime",
]
alternative_features = [
    "sentiment_mean",
    "sentiment_median",
    "max_positive_sentiment",
    "max_negative_sentiment",
    "positive_count",
    "negative_count",
    "neutral_count",
    "article_volume",
    "sentiment_dispersion",
    "positive_news_ratio",
    "negative_news_ratio",
    "sentiment_momentum",
    "sentiment_rolling_7",
    "sentiment_surprise",
    "abnormal_news_volume",
    "extreme_negative_news",
    "sentiment_volume_interaction",
]

model_data = dataset.iloc[30:].reset_index(drop=True)
train_idx, validation_idx, test_idx = chronological_splits(
    len(model_data),
    CONFIG["model"]["validation_fraction"],
    CONFIG["model"]["test_fraction"],
)
target = model_data["target"]
feature_groups = {
    "market_only": market_features,
    "alternative_only": alternative_features,
    "combined": market_features + alternative_features,
}

results = []
fitted_by_group: dict[str, tuple[object, list[str], np.ndarray]] = {}
for group_name, columns in feature_groups.items():
    features = model_data[columns].replace([np.inf, -np.inf], np.nan)
    _, validation_table = fit_compare(
        features, target, train_idx, validation_idx
    )
    selected_name = validation_table.iloc[0]["model"]
    selected_validation = validation_table.iloc[0]

    model = clone(candidate_models(CONFIG["seed"])[selected_name])
    fit_idx = np.concatenate([train_idx, validation_idx])
    model.fit(features.iloc[fit_idx], target.iloc[fit_idx])
    probability = model.predict_proba(features.iloc[test_idx])[:, 1]
    prediction = (probability >= 0.50).astype(int)
    metrics = classification_metrics(
        target.iloc[test_idx], prediction, probability
    )
    interval_low, interval_high = moving_block_accuracy_interval(
        target.iloc[test_idx], prediction, seed=CONFIG["seed"]
    )
    metrics.update(
        {
            "feature_group": group_name,
            "model": selected_name,
            "validation_roc_auc": selected_validation["validation_roc_auc"],
            "validation_pr_auc": selected_validation["validation_pr_auc"],
            "accuracy_ci_low": interval_low,
            "accuracy_ci_high": interval_high,
            "split": "test",
            "n": len(test_idx),
        }
    )
    results.append(metrics)
    fitted_by_group[group_name] = (model, columns, probability)

comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
comparison.to_csv(ROOT / "reports/model_comparison.csv", index=False)

# Alternative-data coverage is too short to support a deployment choice. The
# market-only model is therefore the prespecified final artifact for this run.
final_group = "market_only"
final_model, final_columns, final_probability = fitted_by_group[final_group]
joblib.dump(
    {
        "model": final_model,
        "features": final_columns,
        "feature_group": final_group,
    },
    ROOT / "models/final_model.joblib",
)

# Forecast recorded on t is applied to realised return on t+1.
backtest_frame, backtest_summary = backtest(
    model_data["date"].iloc[test_idx],
    final_probability,
    model_data["daily_return"].iloc[test_idx],
    threshold=CONFIG["model"]["probability_threshold"],
    cost_bps=CONFIG["model"]["transaction_cost_bps"],
    slippage_bps=CONFIG["model"]["slippage_bps"],
)
backtest_frame.to_csv(ROOT / "reports/backtest_daily.csv", index=False)
backtest_summary.to_frame("value").to_csv(
    ROOT / "reports/backtest_summary.csv"
)

save_line_chart("bitcoin_price", "Bitcoin closing price", market["date"], market["close"], "USD")
save_line_chart("volume", "Bitcoin daily volume", market["date"], market["volume"], "Volume")
save_line_chart(
    "rolling_volatility",
    "Lagged 30-day rolling volatility",
    market_with_features["date"],
    market_with_features["rolling_volatility_30"],
    "Daily volatility",
)
save_line_chart("drawdown", "Bitcoin drawdown", market_with_features["date"], market_with_features["drawdown"], "Drawdown")
save_line_chart("news_volume", "Bitcoin news volume", daily_news["date"], daily_news["article_volume"], "Articles")
save_line_chart("daily_sentiment", "Mean daily VADER sentiment", daily_news["date"], daily_news["sentiment_mean"], "Compound score")

figure, axis = plt.subplots(figsize=(8, 5))
axis.hist(market_with_features["daily_return"].dropna(), bins=60)
axis.set(
    title="Bitcoin daily return distribution",
    xlabel="Daily return",
    ylabel="Days",
)
figure.tight_layout()
figure.savefig(ROOT / "reports/figures/return_distribution.png", dpi=140)
plt.close(figure)

figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(backtest_frame["date"], backtest_frame["strategy_equity"], label="Strategy")
axis.plot(backtest_frame["date"], backtest_frame["buy_hold_equity"], label="Buy and hold")
axis.set(title="Out-of-sample backtest", xlabel="Date", ylabel="Growth of $1")
axis.legend()
figure.tight_layout()
figure.savefig(ROOT / "reports/figures/backtest.png", dpi=140)
plt.close(figure)

final_features = model_data[final_columns].replace([np.inf, -np.inf], np.nan)
importance_result = permutation_importance(
    final_model,
    final_features.iloc[test_idx],
    target.iloc[test_idx],
    n_repeats=10,
    random_state=CONFIG["seed"],
    scoring="roc_auc",
    n_jobs=1,
)
importance = pd.DataFrame(
    {
        "feature": final_columns,
        "importance": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }
).sort_values("importance", ascending=False)
importance.to_csv(ROOT / "reports/permutation_importance.csv", index=False)

figure, axis = plt.subplots(figsize=(8, 6))
top_features = importance.head(15).sort_values("importance")
axis.barh(
    top_features["feature"],
    top_features["importance"],
    xerr=top_features["importance_std"],
)
axis.set(
    title="Test-set permutation importance",
    xlabel="ROC-AUC decrease",
    ylabel="Feature",
)
figure.tight_layout()
figure.savefig(ROOT / "reports/figures/permutation_importance.png", dpi=140)
plt.close(figure)

metadata = {
    "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "rows": len(dataset),
    "date_min": str(dataset["date"].min()),
    "date_max": str(dataset["date"].max()),
    "news_rows": len(scored_news),
    "final_feature_group": final_group,
    "final_model": comparison.loc[
        comparison["feature_group"] == final_group, "model"
    ].iloc[0],
    "alternative_training_coverage": float(
        model_data["sentiment_mean"].iloc[train_idx].notna().mean()
    ),
    "test_start": str(model_data["date"].iloc[test_idx].min()),
    "test_end": str(model_data["date"].iloc[test_idx].max()),
}
(ROOT / "reports/run_metadata.json").write_text(json.dumps(metadata, indent=2))

LOGGER.info("Run metadata:\n%s", json.dumps(metadata, indent=2))
LOGGER.info("Model comparison:\n%s", comparison.to_string(index=False))
LOGGER.info("Backtest summary:\n%s", backtest_summary.to_string())
