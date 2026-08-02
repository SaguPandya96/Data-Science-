# Feature dictionary

All market rolling statistics are shifted one day; daily news aggregates are shifted one full day at merge time.

| Feature family | Examples / formula | Intuition | Scaling | Leakage risk/control |
|---|---|---|---|---|
| Returns | `close.pct_change()`, lags 1/2/3/7 | Short-run persistence/reversal | Yes for linear model | Lagged before use |
| Trend | price / lagged MA - 1; momentum | Trend state | Yes | MA excludes current close |
| Risk | lagged rolling return standard deviation | Conditional risk | Yes | Shift before rolling |
| Oscillators | RSI(14), MACD, Bollinger position | Overbought/trend signals | Yes | Lagged inputs where required |
| Liquidity | volume change, lagged z-score | Attention/liquidity | Yes | Reference window is historical |
| Range | (high-low)/open; (close-open)/open | Intraday realized movement | Yes | Known by daily cutoff |
| Sentiment | VADER compound mean/median/dispersion | News tone | Yes | Full-day lag at merge |
| News balance | positive/negative article ratios | Direction of attention | No/optional | Full-day lag |
| News surprise | sentiment minus prior rolling mean | Abnormal tone | Yes | Prior mean shifted |
| News intensity | volume / prior 30-day mean | Abnormal attention | Yes | Denominator shifted |
| Target | `close[t+1]/close[t]-1 > 0` | Next-day direction | Never a feature | Removed before fitting |
