# Modelling dataset dictionary

`data/processed/modelling_dataset.csv` contains one row per UTC date. The table below separates source fields, engineered features, timing fields, and outcomes. Feature formulas and leakage notes are documented in `FEATURE_DICTIONARY.md`.

| Column or family | Type | Meaning |
|---|---|---|
| `date` | UTC timestamp | Market observation date |
| `open`, `high`, `low`, `close` | float | Daily BTC-USD prices |
| `volume` | float | Reported daily trading volume |
| `daily_return`, `log_return` | float | Close-to-close arithmetic and log returns |
| `return_lag_*` | float | Returns shifted by 1, 2, 3, or 7 days |
| `rolling_mean_*` | float | Historical mean return over 7, 14, or 30 days |
| `rolling_volatility_*` | float | Historical return standard deviation over 7, 14, or 30 days |
| `ma_ratio_*` | float | Close relative to its lagged moving average |
| `volume_change`, `volume_zscore_30` | float | Daily and abnormal volume measures |
| `momentum_7`, `momentum_30` | float | Multi-day price momentum |
| `rsi_14`, `macd`, `bollinger_position` | float | Technical trend and oscillator measures |
| `high_low_range`, `open_close_spread` | float | Intraday range and price movement |
| `rolling_skew_30`, `rolling_kurtosis_30` | float | Historical return-shape estimates |
| `drawdown` | float | Close relative to the running historical maximum |
| `volatility_regime` | integer | Causal low/high-volatility indicator |
| `sentiment_*` | float | Daily VADER tone, dispersion, momentum, surprise, and rolling level |
| `positive_count`, `negative_count`, `neutral_count` | integer | Daily headline counts by VADER class |
| `article_volume` | integer | Number of Bitcoin-related headlines assigned to the feature date |
| `positive_news_ratio`, `negative_news_ratio` | float | Class counts divided by article volume |
| `abnormal_news_volume` | float | Article volume relative to its prior rolling mean |
| `extreme_negative_news` | integer | Tone below its prior expanding tenth percentile |
| `sentiment_volume_interaction` | float | Mean sentiment multiplied by article volume |
| `market_quality_flag` | boolean | Structural market-data validation flag |
| `missing_calendar_days_before` | integer | Calendar gap before the market row |
| `feature_timestamp` | UTC timestamp | Latest timestamp represented by the feature row |
| `target_timestamp` | UTC timestamp | Timestamp of the forecast outcome |
| `next_day_return` | float | `close[t+1] / close[t] - 1` |
| `target` | integer | 1 when `next_day_return > 0`, otherwise 0 |

Missing sentiment values mean no usable headline aggregate was available for that date. Count and volume fields are zero-filled; tone fields remain missing and are handled inside each training pipeline.
