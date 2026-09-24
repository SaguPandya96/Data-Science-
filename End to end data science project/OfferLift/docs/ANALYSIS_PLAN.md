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

- Added paired bootstrap intervals to the budget table (listed as a next step in the first
  readout). No model, split, metric or decision rule changed.
- The Bonferroni and alternative-split checks on the 10% budget result were run once,
  after seeing that result. They are reported in the README as follow-up checks and are
  not part of the decision rule.

## Confirmation of the 10% budget result

Written after the first readout and before any confirmation run. Settings are in the
`confirmation` block of `configs/config.yaml`.

**Hypothesis.** At a 10% budget, customers chosen by the logistic T-learner gain more
extra visits from the men's email than the same number chosen at random.

**Why a new design.** The first result came from one 70/30 split, was the best of twelve
cells, and was noticed after the fact. Cross-fitting uses every customer for evaluation
and averages over many splits, so no single lucky split can produce the result.

**Procedure.**
1. Men's email vs no email, outcome `visit`, all 42,613 customers.
2. Split into 5 folds stratified by arm and outcome. For each fold, fit the logistic
   T-learner on the other four and score the held-out fold.
3. Within each fold, target the top 10% by score (scores from different fitted models
   are never ranked against each other).
4. Gain = uplift among targeted customers minus the uplift of the whole population
   (the effect of random targeting), in extra visits per 1,000 customers.
5. For each repeat, a 97.5% bootstrap interval (400 resamples) for the gain and a
   bootstrap p-value.
6. Repeat steps 2–5 with 20 different fold assignments.

**Aggregation** (Chernozhukov, Demirer, Duflo and Fernández-Val, *Generic Machine
Learning Inference on Heterogeneous Treatment Effects*): the point estimate is the
median gain across repeats; the 95% interval uses the median of the 97.5% lower bounds
and the median of the 97.5% upper bounds; the p-value is min(1, 2 × median p-value).

**Pass rule.** The 10% result is confirmed if the aggregated 95% interval is entirely
above zero. If it is, the recommendation changes for budgets of 10% or less only: target
the top-scoring customers. The Qini rule above still governs larger budgets. If it is not,
the recommendation stays random targeting at every budget.

**Caveat, stated in advance.** The customers are the same ones that produced the original
result, so this guards against a lucky split and a lucky model fit, not against something
peculiar to this dataset. Only a new experiment can rule that out.
