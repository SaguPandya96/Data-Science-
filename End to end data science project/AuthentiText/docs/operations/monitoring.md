# Privacy-safe local monitoring

AuthentiText exposes a process-local aggregate snapshot at `GET /v1/metrics`.
The snapshot starts empty whenever the service starts and reflects only actual
requests received by that process. Reading the endpoint does not increment its
own counters.

## Signals

The version 1 schema contains:

- prediction and batch request counts by fixed endpoint and HTTP status;
- errors grouped into fixed application error codes;
- p50, p95, p99, and maximum latency over the most recent 2,048 prediction
  requests;
- successful prediction item counts, three-way category counts, and uncertain
  rate;
- evidence-quality and warning-code counts;
- fixed histograms for character count, whitespace-token count, and calibrated
  machine-likelihood score; and
- service version, model readiness, and verified model/calibration hashes.

The latency window is bounded; request totals and histogram counters cover the
current process lifetime. Quantiles use the nearest-rank method over the current
bounded window. Empty rates and quantiles are returned as `null`, not zero.

## Privacy boundary

The monitor retains no submitted text, excerpts, tokens, features, request or
network identifiers, user agent, raw score, individual calibrated score,
per-request record, or timestamp. It also does not hash text. Unsalted hashes of
submitted text would still enable dictionary matching and would create an
unnecessary persistent identifier, so they are explicitly excluded.

The only input-derived values are added to coarse, process-level aggregate
histograms after a successful prediction. Warning and error labels are mapped
to fixed allowlists; unexpected labels collapse to `other` to prevent
unbounded-cardinality or input-derived labels. There is no database, file
export, remote telemetry, or background sender.

These aggregates are useful for local health and distribution observation, but
they are not a drift detector and do not trigger model changes. Any future
persistence, remote export, or alerting requires a new privacy and retention
decision.
