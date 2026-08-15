# Validation calibration and abstention policy

The word TF-IDF baseline's raw score is mapped to three probabilistic categories
using validation data only. The versioned evidence is in
[`mage_calibration_report.json`](../../data/metadata/mage_calibration_report.json),
and the ignored fitted artifact is `artifacts/baselines/id/calibration_policy.joblib`.

## Reproduce and verify

```powershell
python scripts/calibrate_baseline.py
python scripts/calibrate_baseline.py --verify-only
```

Verification checks the source prediction and calibration artifact hashes,
reloads the artifact, validates its base-model linkage and threshold ordering,
and recomputes calibration, overall policy, and domain policy metrics on the
audit role.

## Three validation roles

Each stable validation record ID is hashed again with SHA-256. The first 64
digest bits modulo 10 assign disjoint roles before calibration:

| Role | Buckets | Rows | Human | Machine | Use |
| --- | --- | ---: | ---: | ---: | --- |
| Calibration fit | 0–3 | 20,135 | 10,216 | 9,919 | Fit sigmoid and isotonic maps |
| Policy selection | 4–6 | 15,304 | 7,658 | 7,646 | Select method and thresholds |
| Calibration audit | 7–9 | 15,070 | 7,734 | 7,336 | Audit the frozen result only |

No published-test data is read.

## Method selection

The primary selection measure is lower Brier score on policy selection, then
lower 15-bin ECE, then method name for a deterministic tie. Isotonic calibration
wins.

| Method | Brier | ECE | Log loss |
| --- | ---: | ---: | ---: |
| Raw | 0.176330 | 0.044139 | 0.522597 |
| Sigmoid | 0.175287 | 0.034925 | 0.520183 |
| Isotonic | 0.174345 | 0.012366 | 0.517463 |

On the untouched calibration-audit role, isotonic improves Brier from 0.177550
to 0.175116 and ECE from 0.049352 to 0.012167. This supports the selected map
within MAGE validation; it does not make the score a universal probability of
authorship.

## Threshold policy

The policy-selection role sets:

- `score <= 0.231884057971`: likely human-written
- `0.231884057971 < score < 0.717391304348`: uncertain
- `score >= 0.717391304348`: likely machine-generated

The upper threshold targets no more than 5% human false-machine decisions on
selection. The lower threshold independently targets no more than 5% machine
false-human decisions. Ties are handled conservatively by the implemented
inclusive boundary rules.

| Measure | Policy selection | Calibration audit |
| --- | ---: | ---: |
| Coverage | 0.399569 | 0.400464 |
| Uncertain rate | 0.600431 | 0.599536 |
| Human false-machine rate | 0.048315 | 0.050685 |
| Machine false-human rate | 0.040413 | 0.045665 |
| Likely-machine precision | 0.899402 | 0.889203 |
| Likely-human negative predictive value | 0.873205 | 0.865839 |
| Decisive accuracy | 0.888962 | 0.879536 |

The audit human false-machine Wilson 95% interval is [0.046016, 0.055801]; the
point estimate slightly exceeds 5%. The 5% values are selection objectives, not
guarantees.

## Domain limits

Audit coverage ranges from 29.5693% on XSum to 58.4059% on CMV. Human
false-machine rates include 12.2632% on HellaSwag, 10.6326% on TL;DR, and
10.2616% on ROCStories. Yelp's machine false-human rate is 20.2484%. A single
global policy therefore remains uneven across domains even with abstention.

The interface must expose these limits and never describe a decisive category
as proof. Short inputs and domains outside the training distribution require
special caution. Published-test evaluation is the next independent check; the
thresholds will not be changed in response to it.
