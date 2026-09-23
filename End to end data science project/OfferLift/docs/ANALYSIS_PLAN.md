# Analysis plan

These choices were fixed in `configs/config.yaml` before the first run, so the readout
cannot be tuned toward a result. Any later change is listed at the bottom with the reason.

## Experiment

Hillstrom MineThatData email test (2008): 64,000 customers who bought in the last
12 months, randomized into three equal arms for a two-week window.

| Arm | Meaning |
| --- | --- |
| `No E-Mail` | control |
| `Mens E-Mail` | email featuring men's merchandise |
| `Womens E-Mail` | email featuring women's merchandise |

## Decision the readout supports

1. Did each email change behavior compared with sending nothing?
2. If the budget only covers part of the list, does a model choosing *who* gets the
   email produce more extra visits than picking customers at random?

## Metrics

- **Primary:** `visit` within two weeks. Chosen because conversion is too rare
  (~0.6% in control) to model per customer at this sample size.
- **Secondary:** `conversion`, `spend`.
- Two treatment arms are compared with control on the primary metric; p-values are
  Holm-adjusted across the two comparisons.

## Validity checks (run before any effect is reported)

- **Sample ratio mismatch:** chi-square against the planned 1/3 split; stop if p < 0.001.
- **Covariate balance:** standardized mean difference of every pre-treatment feature;
  flag anything at or above 0.1.
- **Power:** minimum detectable effect for each metric at the achieved sample size.

## Variance reduction

CUPED on `spend`, using `history` (past-year spend, measured before the email) as the
covariate. Reported next to the unadjusted estimate, never instead of it.

## Uplift modeling

- Contrast: `Mens E-Mail` vs `No E-Mail`, fixed in advance.
- Outcome: `visit`.
- 70/30 split, stratified by arm and outcome, seed 42. The test split is used once.
- Features: pre-treatment only (`recency`, `history`, `mens`, `womens`, `newbie`,
  `zip_code`, `channel`). `history_segment` is dropped because it bins `history`.
- Models: T-learner with logistic regression, T-learner with gradient boosting,
  transformed-outcome gradient boosting.
- **Baseline to beat:** random targeting at the same budget. The metric is the Qini
  coefficient with a bootstrap 95% interval; a model is only called useful if the
  interval excludes zero.

## Decision rule

Recommend model-based targeting only if a model's Qini interval excludes zero **and**
it beats random targeting at the chosen budget. Otherwise the recommendation is to
send the better email to everyone the budget allows, chosen at random.

## Changes after first run

None yet.
