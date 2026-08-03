# Investigation report

## What I wanted to understand

I wanted to see whether a small, inspectable workflow could separate suspicious behavioral changes from ordinary campaign volatility. I deliberately avoided asking the model to pronounce an event “fraud.” The useful decision here is whether a campaign window deserves another look.

## Data I worked with

I streamed **500,000** timestamp-sorted rows from Criteo's public attribution dataset. This local sample covers the first **0.85 days** of the 30-day source and contains **173,410 clicks** and **24,621 conversion-linked impressions**.

The source has no invalid-traffic labels. I left all observed rows unchanged and added **12,470** clearly marked experimental rows across **12 episodes**. Three behaviors were intended as abuse stress tests; a popularity spike was included as a negative control.

After aggregation, the unit of analysis was a campaign in a 30-minute window. There were **15,157 windows** with at least five impressions.

## How I investigated it

1. I loaded the event stream into SQLite and built behavioral features in SQL.
2. I summarized volume, click rate, conversions per click, repeated-user activity, source concentration, click timing, and transformed cost.
3. I split the data chronologically at timestamp **51,660**. Nothing from the later period was used to fit feature scaling or model weights.
4. I compared a robust anomaly percentile with a class-weighted logistic model.
5. I combined them into a cautious hybrid score: 70% supervised probability and 30% anomaly percentile.
6. I converted the score into `monitor`, `review`, `temporary_throttle_candidate`, or `escalate_for_manual_decision`. These are offline recommendations, not actions taken against anyone.

## Results on the held-out time period

| Approach | Precision | Recall | F1 | Average precision | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Robust anomaly baseline | 1.3% | 9.1% | 2.3% | 0.056 | 1.3% |
| Supervised logistic model | 54.5% | 54.5% | 54.5% | 0.469 | 0.1% |
| Hybrid review score | 55.6% | 45.5% | 50.0% | 0.459 | 0.1% |

The hybrid did not beat the supervised model on the held-out period (F1 0.500 versus 0.545). I would not hide that result or promote the more complicated score by default. My next iteration would keep the supervised model as the working candidate and redesign the anomaly component specifically around unseen behavior.

The hybrid review threshold was **0.610**. It surfaced **5 of 11** planted abuse windows and queued **4** windows without a planted abuse label.

The adaptive low-and-slow scenario appeared only in the held-out period. **2 of 8** of its positive campaign-windows crossed the review threshold. The clean popularity-spike control produced **2** queued windows.

That negative-control result is important: unusual volume can be legitimate, and the current model sometimes overreacts to it. I would add campaign-change context and a campaign-specific baseline before trusting a volume-driven alert.

## What caught my attention

The highest-scoring test window without a planted abuse label belonged to campaign `23817046` with a risk score of 0.725. Its evidence was: unusually high impression volume. I would treat this as a lead, not a false accusation: the original data has no ground-truth traffic-quality label.

The model coefficients are not causal explanations. They tell me which standardized features moved the fitted score, while the evidence summary gives a more concrete starting point for review. I would want corroborating information—campaign changes, network or user-agent patterns, historical source behavior, and downstream quality—before taking a restrictive action.

## The decision I would make

I would use this workflow to prioritize a finite review queue. I would not allow the current score to issue a permanent block because:

- the “positive” labels are simulated behaviors;
- unlabeled Criteo traffic is not verified-clean traffic;
- the local sample is an early chronological slice;
- several contextual variables are anonymized;
- the cost field is transformed and cannot be presented as dollars protected.

A reasonable next experiment would replay the scenarios over the full 30 days, hold out entire campaigns rather than only time, and have a second reviewer assess cases without seeing the simulation label. I would also monitor alert volume and feature drift before considering even temporary automated throttling.

## Reproducibility notes

- Random seed: `42`
- Sample SHA-256: `593821593bbe6d66c8f94eec677a8187bc6aa32ff6eea264d5177a782e4984d5`
- Source sampling: first n rows from the timestamp-sorted source
- Source license: CC BY-NC-SA 4.0
- Evaluation labels remain in `evaluation_predictions.csv`; the analyst-facing `review_queue.csv` intentionally hides them.
