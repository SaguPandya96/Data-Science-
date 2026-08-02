# Model Card

## Model details

- **System:** SupplyLens severe-delivery-delay classifier
- **Author:** Sagar Pandya
- **Version:** 0.1.0
- **Model:** logistic regression with class weighting, one-hot categorical encoding, numeric imputation/scaling, and isotonic probability calibration
- **Target:** actual delivery more than seven calendar days after scheduled delivery
- **Unit:** ASN/DN shipment
- **Prediction point:** scheduled-delivery commitment
- **Selected policy:** exact top 20% by stable risk rank per batch

## Intended use

Prioritize shipments for human operational review. The score is decision support and is not suitable for automatic supplier penalties, contracting decisions, clinical decisions, or inventory optimization.

## Training data and temporal design

| Split | Dates | Shipments | Positives | Prevalence |
|---|---|---:|---:|---:|
| Train | 2006-05-02 to 2012-12-31 | 4,681 | 267 | 5.70% |
| Validation | 2013-01-02 to 2013-12-31 | 870 | 83 | 9.54% |
| Final test | 2014-01-03 to 2015-12-31 | 1,479 | 159 | 10.75% |

The final test period was not used to choose the model, calibration method, or review capacity.

## Model selection

| Validation model | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| Prevalence baseline | 0.500 | 0.095 | 0.088 |
| Smoothed supplier-rate rule | 0.644 | 0.138 | 0.088 |
| Logistic regression | 0.746 | 0.198 | 0.161 |
| Histogram gradient boosting | 0.739 | 0.187 | 0.140 |

The advanced model did not exceed logistic regression by the predeclared 0.01 PR-AUC margin. Logistic regression was retained as the simpler recommendation. Expanding-window results also show instability across years, reinforcing the need for drift monitoring.

## Calibration

Isotonic calibration produced validation Brier 0.0793 versus 0.0815 for sigmoid and 0.1615 uncalibrated. Calibration was fitted and selected on the validation year. Its near-zero validation calibration error is optimistic because the same validation labels fit the isotonic mapping; the untouched test calibration error of 0.0360 is the more informative result.

## Final test performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6963 |
| PR-AUC | 0.1661 |
| Precision at top 20% | 0.1892 |
| Recall at top 20% | 0.3522 |
| F1 at top 20% | 0.2462 |
| Lift at top 20% | 1.7598× |
| Brier score | 0.0956 |
| Calibration error | 0.0360 |

The policy reviews 296 of 1,479 test shipments and captures 56 of 159 severe delays. The confusion matrix is 1,080 true negatives, 240 false positives, 103 false negatives, and 56 true positives. Isotonic ties are broken by stable descending risk rank so capacity remains exact.

## Explainability

Global explanations use held-out permutation importance on PR-AUC. Local examples use a feature-perturbation diagnostic: one input at a time is replaced by its training reference value and the calibrated score change is reported. These are predictive associations, not causal attributions. The intervention queue includes local contributors for the top 25 ranks; other rows are labeled not computed.

## Segment and error analysis

`reports/tables/segment_performance.csv` reports sample size, positive rate, precision, recall, false-negative rate, Brier score, and a minimum-volume reliability flag for supplier, supplier volume, destination, mode, product group, fulfillment path, value band, calendar quarter, and seen/unseen supplier. `error_examples.csv` preserves high-risk, false-positive, and false-negative examples.

## Lead-time experiment

The scheduled lead-time baseline achieved 0.70 test MAE days. The learned P50 model achieved 4.44 days, and historical median baselines were substantially worse. The learned lead-time model is not recommended. The P90 model attained 90.42% empirical coverage with 1.82 pinball loss and a median 6.21-day P50–P90 width; it remains experimental.

## Limitations and risks

- Data covers a historical public-health supply program and may not represent current operations.
- Target prevalence rises from 5.70% in training to 10.75% in test.
- The exact schedule-entry timestamp is unavailable; target-derived historical aggregates are excluded.
- Freight, weight, insurance, manufacturing site, and mode require prediction-time verification in another system.
- Test PR-AUC is modest and false negatives remain material.
- Supplier patterns can encode structural or geographic context and must not be treated as supplier fault.
- No intervention outcomes exist, so the model does not estimate action effectiveness.

## Ethical and responsible use

Use the queue to focus investigation, preserve human review, show low-volume uncertainty, and monitor segment behavior. Do not use model scores as automatic evidence of negligence or to deny access to suppliers, regions, or product groups.

