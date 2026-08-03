# Model Card

## Intended use

Support daily store-level sales planning, portfolio forecast review, and sensitivity analysis. The output can inform human review of staffing, inventory, and financial plans when combined with operational constraints and domain judgment.

## Model and baselines

- naive baseline: prior seven-observation rolling average;
- interpretable baseline: linear regression after median imputation and one-hot encoding;
- default production candidate: XGBoost regression with the same fitted preprocessing contract;
- CI/sample candidate: a small random forest that validates plumbing, not business accuracy.

The final six weeks form a chronological holdout. Metrics include RMSE, MAE, WAPE, forecast bias, and R-squared.

## Verified full-data result

The default XGBoost configuration trained on 970,379 rows and was evaluated on 46,830 store-days from 2015-06-20 through 2015-07-31. It achieved RMSE 973.81, MAE 630.99, WAPE 10.51%, forecast bias 1.65%, and R² 0.932. The exact reference tables are checked into `reports/reference/`.

## Important controls

- customer count is excluded to avoid hindsight leakage;
- all sales lags are shifted within store before rolling;
- preprocessing is fitted inside the training pipeline;
- unseen categories are ignored safely during one-hot encoding;
- predictions are clipped at zero;
- model, run metadata, error segments, and training distributions are persisted together.

## Limitations

- The default holdout measures rolling one-day-ahead performance, not a one-shot six-week horizon.
- Scenario differences reflect model associations and are not causal treatment effects.
- Store identity may dominate importance without representing an actionable business lever.
- Public-source data is historical, anonymized, and may not represent current operations.
- The target is turnover and does not include margin, costs, cannibalization, stockouts, or inventory constraints.
- No prediction intervals are included in version 1.
- Cold-start lag values use zero and should be replaced with a governed prior for new stores.
- The implemented scoring contract rejects stores without history and plans containing multiple future dates.

## Out-of-scope use

Do not use the model as an autonomous promotion-allocation engine, as evidence of causal ROI, or as the sole input to employment, credit, pricing, or other high-impact decisions.

## Review gates

Before deployment, require source approval, checksum-pinned training data, horizon-appropriate backtests, comparison with the current business forecast, prediction intervals, segment acceptance criteria, owner sign-off, and rollback procedures.
