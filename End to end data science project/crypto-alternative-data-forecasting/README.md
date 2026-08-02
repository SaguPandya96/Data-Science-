# Bitcoin Direction Forecasting with News Sentiment

This repository documents a simple question that turned out to have an inconvenient answer: does adding Bitcoin-news sentiment improve a next-day direction model built from market data?

For the data available here, it does not. The market model was below chance on the held-out period, the combined model did not improve it, and the available GDELT history was too short to support a serious multi-year sentiment test. I kept that result rather than tuning until the conclusion changed.

## What is being predicted?

Each row represents one UTC day. The target is:

```text
target[t] = 1 if close[t+1] > close[t], otherwise 0
```

This is a classification problem rather than an exact-price forecast. Direction maps more directly to the long/cash decision used in the backtest, avoids modelling Bitcoin's nonstationary price level, and produces probabilities that can be filtered with a trading threshold.

The target and the trading position are not the same thing. A position also depends on when the forecast becomes available, when it is executed, the chosen probability threshold, and trading costs.

## Data used

The checked-in run contains:

- 1,642 daily BTC-USD market observations from 2022-01-01 through 2026-06-30;
- 304 Bitcoin-related GDELT headlines from the rolling window available at collection time;
- headline-level VADER sentiment scores; and
- a merged daily modelling table with causal market and news features.

Yahoo Finance supplies OHLCV data. GDELT DOC 2.0 supplies headline metadata without an API key. GDELT's `seendate` is an observation timestamp, not a guarantee of first publication, and DOC 2.0 is a rolling search service rather than a complete historical archive. Those details matter for any point-in-time claim. [DATA_SOURCES.md](DATA_SOURCES.md) records the source-specific caveats.

## Research design

The study uses a chronological train/validation/test split. Candidate models are selected on validation data; the held-out test set is used only for final reporting. Scaling and imputation are fitted inside each training pipeline. Market rolling windows are shifted, daily news features receive a full-day lag, and neither `target` nor `next_day_return` is passed to a model.

The main comparisons are:

1. majority class, previous-day direction, and a moving-average rule;
2. logistic regression and a constrained random forest;
3. market-only features;
4. alternative-data-only features; and
5. combined market and alternative features.

The final saved artifact is deliberately the market-only model. The sentiment history covers too little of the training period to justify selecting a combined model for use.

## Results from the checked-in run

| Feature set | Selected model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Market only | Logistic regression | 0.4675 | 0.4850 | 0.4607 | 0.4409 |
| Combined | Logistic regression | 0.4675 | 0.4850 | 0.4607 | 0.4409 |
| Alternative only | Logistic regression | 0.4706 | 0.6400 | 0.5000 | 0.4706 |

The alternative-only result is a constant, no-information forecast. Most historical folds contain no sentiment observations, so imputation cannot turn the rolling GDELT sample into a historical signal. Combined results match market-only results for the same reason.

These numbers do not support a predictive edge. The block-bootstrap accuracy interval is reported in `reports/model_comparison.csv`, but an interval around a weak estimate does not repair the underlying coverage problem.

The long/cash backtest uses yesterday's signal for today's realised close-to-close return, a 0.55 entry threshold, 10 basis points of transaction cost, and 5 basis points of slippage. It lost 40.1%, recorded a −1.45 Sharpe ratio, and made 76 position changes in the held-out period. The weak classifier therefore failed economically as well as statistically.

## Feature set

Market variables include lagged returns, rolling means and volatility, volume change and z-score, momentum, RSI, MACD, Bollinger position, moving-average ratios, daily range, open-close spread, skewness, kurtosis, drawdown, and a causal volatility regime.

News variables include mean and median sentiment, positive and negative ratios, article volume, abnormal volume, dispersion, momentum, surprise, rolling sentiment, extreme-negative-news flags, and sentiment-volume interaction.

The full formulas, intuition, scaling notes, and leakage controls are in [data/FEATURE_DICTIONARY.md](data/FEATURE_DICTIONARY.md).

## Repository layout

```text
data/
  raw/          source snapshots
  interim/      cleaned market data and scored headlines
  processed/    final modelling table
models/         fitted final model bundle
notebooks/      complete research notebook
reports/        measured tables and figures
src/            ingestion, cleaning, features, models, evaluation and backtest
tests/          chronology, leakage, aggregation and execution tests
```

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
pytest -q
jupyter notebook notebooks/crypto_forecasting_end_to_end.ipynb
```

Raw downloads are cached. A normal rerun uses the existing source snapshots; remove a specific raw file only when an intentional refresh is required.

## What the notebook contains

The notebook shows the work rather than only loading final tables:

- market and news data-quality checks;
- separate price, volume, return, volatility, drawdown, autocorrelation, correlation, regime, news-volume, and sentiment charts;
- feature formulas and point-in-time assertions;
- chronological splits and expanding-window evaluation;
- baseline, logistic-regression, and random-forest training;
- ROC, precision-recall, confusion-matrix, and permutation-importance analysis;
- market-only, sentiment-only, and combined comparisons;
- false-positive and false-negative analysis;
- execution-lagged backtesting and cost sensitivity; and
- limitations and a concrete next-data plan.

## Limitations

The largest weakness is not the classifier; it is the alternative-data history. A credible incremental-value test needs point-in-time headlines across every training and evaluation period. Other limitations include possible vendor revisions, ambiguity between GDELT observation time and publication time, VADER's general-language lexicon, one asset, dependent daily observations, changing market regimes, and the risk of learning from repeated test-set inspection.

The next version should start with a licensed or archived headline history, a manually labelled sentiment audit, cached FinBERT scores, preregistered robustness choices, and block-based uncertainty estimates across multiple assets. Until then, the defensible conclusion is that this run did not show useful forecast or trading value.

## Disclaimer

This repository is an educational research study, not investment advice. Historical results do not establish live tradability or future performance.
