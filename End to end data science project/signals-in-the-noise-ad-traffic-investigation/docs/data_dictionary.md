# Data dictionary

## Source fields used

| Field | Meaning in the public dataset | How I used it |
|---|---|---|
| `timestamp` | Seconds from the first impression | Chronological split and 30-minute windows |
| `uid` | Anonymized user identifier | Unique-user and repeat-event summaries |
| `campaign` | Anonymized campaign identifier | Main investigation entity |
| `click` | Whether the impression was clicked | CTR, click timing, and concentration |
| `conversion` | Whether a conversion occurred within 30 days | Downstream-quality context |
| `cost` | Transformed display cost | Relative exposure in cost units; never dollars |
| `time_since_last_click` | Seconds since the user's previous click | Timing context |
| `cat1`, `cat2` | Anonymized contextual categories | Combined into a stable `source_id` grouping |

`source_id` is my own deterministic grouping of two anonymous context columns. It is not a publisher identity and should not be described as one.

## Campaign-window features

| Feature | Definition | Why I kept it |
|---|---|---|
| `impressions` | Events in the campaign's 30-minute window | Volume changes are useful but ambiguous |
| `clicks` | Clicked impressions | Separates exposure from interaction |
| `ctr` | Clicks / impressions | Surfaces unusually high or low interaction |
| `conversion_rate` | Conversion-linked impressions / impressions | Outcome modeled in the observed-only quality track |
| `conversion_per_click` | Conversions / clicks | Adds downstream context without treating non-conversion as proof |
| `repeat_event_share` | Share of events from user IDs seen more than once in the window | Captures concentration among returning IDs |
| `top_source_share` | Share of events from the largest anonymous source group | Captures source concentration |
| `active_click_minute_share` | Minutes with a click / 30 | Distinguishes bursts from spread-out activity |
| `clicks_per_unique_user` | Clicks / unique users | Another view of user concentration |
| `total_cost_units` | Sum of transformed cost | Relative exposure only |

## Campaign-history fields

These fields are calculated only for windows with no simulated events. Each rolling value uses the campaign's previous 20 available windows and excludes the current window.

| Field | Definition |
|---|---|
| `history_windows` | Number of earlier campaign windows available, up to 20 |
| `baseline_impressions` | Mean impressions across those earlier windows |
| `baseline_ctr` | Mean click rate across those earlier windows |
| `baseline_conversion_rate` | Mean conversion-linked impression rate across those earlier windows |
| `baseline_cost_units` | Mean transformed cost units across those earlier windows |
| `volume_change_ratio` | Current impressions divided by the earlier campaign mean |
| `ctr_change_from_baseline` | Current CTR minus the earlier campaign mean |
| `conversion_change_from_baseline` | Current conversion rate minus the earlier campaign mean |

## Observed expected-behavior fields

| Field | Definition | How I interpret it |
|---|---|---|
| `expected_ctr` | Click probability estimated from the earlier training period | A forecast, not a quality judgment |
| `expected_clicks` | Impressions multiplied by expected CTR | Direct comparison with actual clicks |
| `expected_conversion_rate` | Estimated probability of a conversion-linked impression | A downstream forecast |
| `expected_conversions` | Impressions multiplied by expected conversion rate | Direct comparison with actual conversion-linked impressions |
| `click_deviation_z` | Click residual divided by expected binomial variation | Positive means more clicks than expected; negative means fewer |
| `conversion_deviation_z` | Conversion residual divided by expected binomial variation | A large negative value is a downstream shortfall |
| `quality_risk_score` | Empirical training-period percentile of the combined residual score | Review priority only |
| `recommended_action` | `monitor` or `review_traffic_quality` | No automatic restriction is supported |
| `evidence_summary` | Plain-language description of the strongest contributing signals | Starting hypothesis for a reviewer |

## Evaluation-only fields

`is_simulated_abuse`, `scenario`, and `episode_id` exist only to evaluate the stress test. They are excluded from model features and hidden from `review_queue.csv`.

They are also absent from `observed_review_queue.csv`. In addition, every campaign-window containing a simulated row is removed before the observed expected-behavior models are fitted or scored.
