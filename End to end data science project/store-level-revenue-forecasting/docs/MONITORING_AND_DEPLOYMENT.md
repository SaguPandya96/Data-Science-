# Monitoring and Deployment

## Batch deployment pattern

1. Validate that sales actuals are complete through the declared cutoff.
2. Load approved store metadata and future-known operating, holiday, and promotion inputs.
3. Build shifted lag features and score the saved pipeline.
4. Check row counts, missingness, category coverage, and non-negative output.
5. Publish forecasts with model version, forecast origin, target date, and scenario name.
6. Join actuals later and calculate rolling accuracy and bias.

The serialized joblib file must only be loaded from a trusted build. Treat it as executable code, retain the training environment lock, and roll back by model version rather than overwriting artifacts in place.

## Monitoring signals

| Signal | Suggested review |
|---|---|
| source freshness and expected store/date coverage | every run |
| missing values and unseen categories | every run |
| prediction totals and zero/negative rate | every run |
| WAPE and forecast bias overall | daily with weekly trend |
| WAPE and bias by store, promotion, and weekday | weekly |
| numeric feature mean shift relative to `monitoring_baseline.json` | weekly |
| categorical frequency and unseen-level change | weekly |
| scenario total changes | when promotion plans change |

Thresholds should be calibrated from backtests and business tolerance. A practical starting review rule is an absolute four-week bias above 5%, WAPE deterioration above 20% relative to the accepted backtest, missing expected stores, or input distributions outside their historical ranges. These are review triggers, not universal performance guarantees.

## Retraining and rollback

Retrain on a schedule only after data completeness checks. Compare the challenger with both the current model and business baseline on multiple temporal windows. Promote only when acceptance criteria pass, then retain the prior model and configuration for immediate rollback.

## Scenario governance

Promotion and demand scenarios should carry the model version and assumptions used. Finance outputs must label them as model-estimated sensitivities. Causal claims require randomized experiments or a separate causal design with overlap, timing, and confounding diagnostics.
