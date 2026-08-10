# Baseline model training

Three first-cycle baselines are trained only on the 287,843 sanitized MAGE
train records. Validation and published-test data are not loaded by the
training function. Model files remain ignored under `artifacts/baselines/id/`;
their configurations and byte identities are versioned in
[`mage_baseline_training_report.json`](../../data/metadata/mage_baseline_training_report.json).

## Reproduce and verify

```powershell
python scripts/train_baselines.py
python scripts/train_baselines.py --verify-only
```

The verification path checks every artifact's size and SHA-256, reloads it,
confirms its declared model type, and requires finite `[0, 1]` scores from two
synthetic smoke inputs. It does not retrain.

## Models

1. **Majority:** always predicts the training majority class. Its score is the
   measured machine-positive training prevalence, 0.697911709.
2. **Length logistic diagnostic:** balanced logistic regression over
   `log1p(characters)` and `log1p(whitespace tokens)`, with standardized
   features. This measures a known dataset artifact and is not a product
   candidate.
3. **Word TF-IDF logistic:** balanced logistic regression over lowercased word
   unigrams and bigrams. TF-IDF uses sublinear term frequency, L2 normalization,
   `min_df=5`, `max_df=0.995`, float32 values, and at most 100,000 features. The
   classifier uses SAGA, `C=1`, tolerance 0.001, at most 50 iterations, and seed
   1729.

No source, domain, generator, partition, target, content hash, or record ID is
provided as a model feature.

## Executed run

The training input contains 86,954 human and 200,889 machine records. The word
TF-IDF matrix has 287,843 rows, 100,000 features, and 55,726,925 nonzero values
(density 0.001936018). Vectorization took 141.768 seconds and classifier fitting
took 16.545 seconds on the documented workstation. The classifier stopped after
16 iterations with no convergence warning. The length model fitted in 4.616
seconds and stopped after 13 iterations. Timings are observations from this
machine, not runtime guarantees.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `majority.joblib` | 106 | `036b496c23f325ada27f2f173677eabb494dad9ba2541ae0ff9b5a615219f992` |
| `length_logistic.joblib` | 928 | `a181b668e26b09c15c1968422e4ee442a48f097acb4bf61aa0a59d22899c3705` |
| `word_tfidf_logistic.joblib` | 1,578,216 | `474481ed4548f89c3d0308bb8389f114f1e1d55a4436c300e3e9fb08d4eb45dd` |

Artifact compression makes disk size much smaller than the transient sparse
training matrix. Peak memory was not instrumented, so no peak-memory claim is
made.

## Scope

This stage proves training reproducibility, bounded configuration, convergence,
and reloadability. It reports no discrimination, calibration, error, or
generalization result. Those measurements begin on validation data in the next
stage, before any published-test evaluation.
