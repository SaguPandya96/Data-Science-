# Model card

## Intended use

Prioritize campaign-time windows for manual investigation in this offline experiment.

## Out-of-scope use

This model should not identify people, label original Criteo traffic as fraudulent, estimate real monetary loss, or take permanent enforcement actions.

## Model

- Supervised component: class-weighted logistic regression implemented with NumPy
- Anomaly component: robust feature distance converted to an empirical percentile
- Combination: `0.70 * supervised_score + 0.30 * empirical_anomaly_percentile`
- Chronological test cutoff: `51660` seconds
- Review threshold: `0.6100`

## Held-out experimental performance

- Precision: 55.6%
- Recall: 45.5%
- Average precision: `0.459`
- False-positive rate: 0.1%

These figures measure recovery of planted experimental scenarios. They are not estimates of real-world fraud-detection performance.

## Coefficients

Coefficients are based on standardized features. Magnitude indicates association with the fitted score, not causality.

| Feature | Coefficient |
|---|---:|
| `log_impressions` | 0.6100 |
| `ctr` | -0.4342 |
| `repeat_event_share` | 0.4160 |
| `clicks_per_unique_user` | 0.1918 |
| `top_source_share` | -0.1026 |
| `log_total_cost_units` | 0.1007 |
| `log_clicks` | 0.0411 |
| `active_click_minute_share` | 0.0340 |
| `conversion_per_click` | -0.0276 |

## Known risks

- Simulation-to-reality gap
- Unverified negative class
- Early-period sampling bias
- Possible campaign-specific memorization
- Anonymized contextual features
- Threshold instability under changing traffic

## Safeguards

- Human review is the default intervention.
- Simulation labels are hidden from the operational review queue.
- Every alert contains a plain-language evidence summary.
- Restrictive actions are described as candidates, not automatically executed.
- Permanent blocking is not supported.
