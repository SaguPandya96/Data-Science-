# SupplyLens Engineering Notes

I keep this file focused on decisions that materially changed the project rather than on every command used during development.

## Exact-capacity queue selection

Isotonic calibration created tied probabilities. My first threshold-based implementation could therefore select more shipments than the configured review capacity. I replaced it with stable descending ranks so scoring and evaluation always select the same exact number of rows. Automated tests cover the tie behavior.

## Prediction-time boundary

The source does not provide a trustworthy timestamp for when every field became available. I treated scheduled-delivery commitment as the scoring point, excluded delivery outcomes, and left target-derived history out of the production features when its availability could not be guaranteed.

## Model and calibration choice

Logistic regression remained the selected classifier because histogram gradient boosting did not exceed its validation PR-AUC by the predeclared 0.01 margin. Isotonic calibration had the best validation Brier score, while untouched-test calibration metrics remain the more credible performance evidence.

## Review policy

The operating rule selects the top 20% of each batch because the decision is constrained by review capacity. Results at 5%, 10%, 20%, and 30% are preserved so the tradeoff can be revisited when capacity changes.

## Negative lead-time result

The learned P50 lead-time model underperformed the scheduled lead-time baseline on the final period. I retained that result and did not recommend the learned model for deployment.

## Reproducibility checks

The source download is commit-pinned and checksum-verified. README metrics are generated from `reports/metrics/final_metrics.json`, the canonical notebook is stored with executed outputs, and automated checks cover data validation, temporal splitting, leakage controls, scoring contracts, linting, and tests.
