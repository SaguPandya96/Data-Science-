# Scoring Contract

## Decision being scored

Version 1 scores one future calendar date for one or more stores. The command combines:

- a trusted fitted pipeline;
- actual sales history through the prior day;
- one row of store metadata per store; and
- a future operating-plan CSV.

The forecast date must be later than every historical date. Every forecast store must exist in both history and store metadata.

## Future operating-plan fields

| Field | Meaning |
|---|---|
| `Store` | store identifier present in history and metadata |
| `Date` | the single future forecast date |
| `Open` | planned operating status |
| `Promo` | planned standard promotion status |
| `StateHoliday` | known public-holiday category or `0` |
| `SchoolHoliday` | known school-holiday indicator |

`DayOfWeek` is calculated from `Date`. `Sales` and `Customers` do not belong in the future plan.

## Command

```bash
python score_forecast.py \
  --config config.yaml \
  --future path/to/future_plan.csv \
  --output data/processed/future_predictions.csv
```

Optional `--history`, `--stores`, and `--model` arguments override their configured defaults.

The output contains `Date`, `Store`, and non-negative `PredictedSales`.

## Why multiple dates are rejected

Historical lag features can use actuals only through the forecast origin. Scoring several future dates in one pass would either leak future actuals or silently reuse stale lags. A multi-day product needs an explicit recursive loop, direct horizon-specific models, or frozen-origin features with separate backtests.

## Artifact trust

Joblib artifacts can execute code when loaded. Load only models produced by the trusted build pipeline, retain the matching dependency lock and configuration, and version artifacts rather than overwriting the only rollback copy.
