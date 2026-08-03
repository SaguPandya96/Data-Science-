# Observed traffic quality investigation

## The question I asked

After building the controlled stress test, I wanted a second analysis that did not depend on planted examples at all:

> Using earlier campaign history and the window's non-outcome context, were its clicks and conversion-linked impressions close to what I would have expected?

This is the part of the project I would use to discover leads in the public data. It does not assign a fraud label. A large residual tells me that the model was surprised; it does not tell me why.

## The data I kept

I started with **500,000** source rows covering the first **0.85 days** of Criteo's timestamp-sorted file. I removed every campaign-window touched by my simulation layer before fitting this analysis. That left **15,131 observed-only campaign windows**.

The source has click and conversion outcomes, but it has no invalid-traffic truth, device evidence, network identifiers, or policy decisions. I treated those missing fields as a boundary, not something a model could fill in.

## How I worked through it

1. I built 30-minute campaign summaries in SQL.
2. For each campaign, I calculated rolling baselines from its previous 20 available windows. The current and future windows were excluded.
3. I split the sample chronologically at timestamp **51,660**. The first **9,229** windows were used to fit expected behavior; the remaining **5,902** windows were kept for replay.
4. I fitted one binomial model for click rate and another for conversion-linked impression rate. The second model can use the clicks already observed in that window because it is answering a later-stage quality question.
5. I converted click and conversion residuals into standardized surprises.
6. I combined those surprises with campaign-relative volume, user/source concentration, and transformed cost exposure. Each part was converted to a percentile against the earlier training period.
7. I queued only held-out windows with a quality-risk score at or above **0.990**, a training-period percentile cutoff of 99.0%. The queue is for investigation, not enforcement.

## Did the expected-behavior models travel forward in time?

| Held-out check | Click-rate model | Conversion-rate model |
|---|---:|---:|
| Observed rate | 35.00% | 4.82% |
| Mean expected rate | 35.60% | 4.97% |
| Weighted mean absolute error | 0.06376 | 0.02655 |
| Weighted Brier score | 0.007769 | 0.001535 |
| Weighted log loss | 0.63122 | 0.17549 |

These are calibration and forecasting checks, not detection accuracy. There are no real abuse labels with which to calculate precision or recall.

## What reached the queue

The held-out replay produced **11 review cases** after applying the percentile cutoff and a maximum queue size of **30**.

The first case I would open is campaign `2946551` at timestamp `64800`. It reached a quality-risk percentile of **0.995** because clicks were higher than the expected range.

I would read each evidence summary as a starting hypothesis. Before any restrictive decision, I would want campaign-change context, longer history, network and user-agent evidence, neighboring-window persistence, and an innocent explanation check.

## What I learned

Campaign history made the investigation more useful than a global outlier score. The same click rate can be routine for one campaign and surprising for another. Residuals also separated two questions that are easy to blur together: whether interaction was unusual, and whether downstream outcomes were weaker than expected after that interaction.

The remaining weakness is the short early-period sample. Some campaigns have little history, and the chronological replay does not cover the source's full 30 days. I would next run the entire dataset, add day-of-week effects, and measure whether the queue stays stable as traffic changes.

## Decision boundary

- An unexpected window is not automatically harmful.
- A weak conversion result is not proof that a click was invalid.
- The score prioritizes review; it does not block, throttle, or accuse anyone.
- Criteo's cost field is transformed, so the project reports cost units rather than dollars.
- Performance from the separate controlled stress test should not be presented as real-world fraud performance.
