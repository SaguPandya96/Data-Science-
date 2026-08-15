# Frozen in-distribution test evaluation

This is the single published-test evaluation of the first-cycle word TF-IDF
baseline, isotonic calibrator, and abstention thresholds. All artifacts were
frozen before scoring. The aggregate report is
[`mage_frozen_test_report.json`](../../data/metadata/mage_frozen_test_report.json).

The ignored prediction file contains stable record metadata, raw and calibrated
scores, and categories—but no text. It has 50,567 rows, 2,811,441 bytes, and
SHA-256
`93306f231655bccf7876f40acc7c1c51901ad8044aba15d8d3db482e2bdfd9ec`.

## Verify

```powershell
python scripts/evaluate_frozen_test.py --verify-only
```

The non-verify command is the one-time scoring path and should not be rerun as a
tuning loop. Verification checks upstream artifacts, prediction identity,
replays isotonic calibration and category assignment, and recomputes every
metric and subgroup from the saved predictions.

## Raw discrimination

The raw word-model score reaches ROC AUC 0.806435 and average precision
0.821832. At the rejected binary threshold 0.5, balanced accuracy is 0.723534,
recall 0.712028, and human false-positive rate 0.264961. The raw Brier score is
0.180328. Validation ROC AUC was 0.814070, so the held-out result is 0.007635
lower.

These are MAGE in-distribution results. They do not establish performance on a
new dataset, generator family, or writing context.

## Frozen calibrated policy

Isotonic-calibrated test Brier score is 0.177779, log loss 0.527602, and 15-bin
ECE 0.009833. Using the frozen 0.231884057971 and 0.717391304348 thresholds:

| Measure | Test result |
| --- | ---: |
| Likely human | 9,685 |
| Uncertain | 28,859 |
| Likely machine | 12,023 |
| Coverage | 0.429292 |
| Uncertain rate | 0.570708 |
| Human false-machine rate | 0.052391 |
| Machine false-human rate | 0.059880 |
| Likely-machine precision | 0.888297 |
| Likely-human negative predictive value | 0.845844 |
| Decisive accuracy | 0.869357 |

There are 1,343 human false-machine decisions among 25,634 human records; the
Wilson 95% interval is [0.049730, 0.055187]. There are 1,493 machine
false-human decisions among 24,933 machine records; the interval is
[0.057003, 0.062894]. Both point estimates miss their 5% selection objectives.
They are reported unchanged.

## Domain variation

| Domain | Coverage | Human false-machine | Machine false-human | Decisive accuracy |
| --- | ---: | ---: | ---: | ---: |
| CMV | 0.618048 | 0.016250 | 0.026704 | 0.965063 |
| ELI5 | 0.450165 | 0.015526 | 0.103912 | 0.866737 |
| HellaSwag | 0.348408 | 0.137303 | 0.010478 | 0.781095 |
| ROCStories | 0.338290 | 0.087462 | 0.018519 | 0.842033 |
| Scientific generation | 0.356725 | 0.025611 | 0.038667 | 0.911007 |
| SQuAD | 0.490008 | 0.006778 | 0.042468 | 0.949837 |
| TL;DR | 0.418525 | 0.097041 | 0.009419 | 0.870859 |
| XSum | 0.356892 | 0.053305 | 0.042717 | 0.865409 |
| Yelp | 0.547028 | 0.005279 | 0.254417 | 0.767229 |

The aggregate policy hides severe asymmetric domain errors. Most notably,
13.7303% of HellaSwag humans receive likely-machine decisions, while 25.4417%
of Yelp machine records receive likely-human decisions.

## Length effects

Below 50 whitespace tokens, coverage is 0.259388, human false-machine rate
0.096887, machine false-human rate 0.119406, and decisive accuracy 0.588441.
For 129–512 tokens, the corresponding rates are 0.551120, 0.019883, 0.025244,
and 0.959341. Inputs under 50 whitespace tokens therefore require an explicit
low-evidence warning even when the score crosses a threshold.

## Runtime and conclusion

Batch scoring and calibration took 15.528 seconds on the documented CPU, or
3,256.566 records/second. This is not request-level latency.

The baseline is useful as an honest research reference and local prototype, not
as reliable authorship evidence. Its high abstention rate, domain asymmetry,
short-text failure, and residual decisive errors must remain visible in the
inference contract and interface. No test-driven retuning is permitted.
