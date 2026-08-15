# Development-reference drift checks

AuthentiText compares process-local aggregate prediction traffic with a frozen,
validation-only reference at `GET /v1/drift`. A result is an investigation flag,
not evidence of authorship, model failure, or a reason to retrain automatically.

## Reference construction

The deterministic builder reads the 50,509 sanitized validation records, the
stored text-free word TF-IDF validation predictions, and the frozen isotonic
calibrator. It verifies every source file's recorded byte size and SHA-256,
matches record IDs, sources, targets, and token counts, and retains only
aggregate distributions in the committed reference. Of those rows, 33,647 are
assigned to reference/threshold windows and 16,862 to disjoint audit windows.
The published test partition is not read.

```powershell
python scripts/build_drift_reference.py
python scripts/build_drift_reference.py --verify-only
```

The reference uses the same character, whitespace-token, calibrated-score, and
three-way category buckets as `GET /v1/metrics`. It assigns validation records
to 60 deterministic windows with the first 64 bits of `sha256(record_id)`:

- windows 0–39 form the runtime reference and select one total-variation
  threshold per signal using the maximum leave-one-window-out distance;
- disjoint windows 40–59 audit same-distribution false alerts; and
- records from those same held-out audit windows are grouped into nine actual
  validation domains and evaluated as documented shifts after thresholds are
  fixed.

The smallest hash window contains 760 rows, so drift status remains
`insufficient_data` below 760 successful operational prediction items.

## Measured backtest

The generated reference records these actual results:

| Check | Result |
| --- | ---: |
| Disjoint same-distribution audit windows flagged | 1 / 20 |
| Measured audit false-alert rate | 5.0% |
| Real validation-domain shifts flagged | 9 / 9 |
| Measured domain-shift sensitivity | 100.0% |

The selected total-variation thresholds are 0.052239784889 for characters,
0.048325451015 for whitespace tokens, 0.070703386134 for calibrated scores, and
0.045813992952 for categories.

These figures are a backtest on one development corpus, not production error
bounds. The domain subsets are record-disjoint from the reference but still come
from the same development corpus, so 9/9 is not an external-generalization
estimate. Even the measured 5% audit false-alert rate is only 1 event among 20
windows, so a flag is phrased as `investigate` and cannot trigger automatic
action.

## Runtime contract

The endpoint verifies that the reference's base-model and calibration hashes
match the loaded predictor. It then returns one of:

- `insufficient_data` before 760 successful prediction items;
- `within_reference` when no signal exceeds its threshold; or
- `investigate` when one or more aggregate signals exceed their threshold.

The response identifies each signal's measured distance, fixed threshold, and
flag. Monitoring remains process-local and cumulative; restarting the service
starts a new observation window. There is no persistence, background export,
alert delivery, automatic threshold update, or automatic retraining.

This version does not claim semantic, language, embedding, adversarial-edit, or
longitudinal drift coverage. Adding those signals requires real reference data,
privacy review, and independent false-alert and sensitivity evaluation.
