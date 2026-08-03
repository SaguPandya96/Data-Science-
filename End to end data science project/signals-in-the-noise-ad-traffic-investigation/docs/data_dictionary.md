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
| `conversion_per_click` | Conversions / clicks | Adds downstream context without treating non-conversion as proof |
| `repeat_event_share` | Share of events from user IDs seen more than once in the window | Captures concentration among returning IDs |
| `top_source_share` | Share of events from the largest anonymous source group | Captures source concentration |
| `active_click_minute_share` | Minutes with a click / 30 | Distinguishes bursts from spread-out activity |
| `clicks_per_unique_user` | Clicks / unique users | Another view of user concentration |
| `total_cost_units` | Sum of transformed cost | Relative exposure only |

## Evaluation-only fields

`is_simulated_abuse`, `scenario`, and `episode_id` exist only to evaluate the stress test. They are excluded from model features and hidden from `review_queue.csv`.
