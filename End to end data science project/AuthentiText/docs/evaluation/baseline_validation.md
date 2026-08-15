# Baseline validation evaluation

This report evaluates the three fixed baseline artifacts on all 50,509
sanitized MAGE validation records. It does not inspect or score the published
test partition. The complete aggregate results are in
[`mage_baseline_validation_report.json`](../../data/metadata/mage_baseline_validation_report.json).

## Reproduce and verify

```powershell
python scripts/evaluate_baselines.py
python scripts/evaluate_baselines.py --verify-only
```

Evaluation writes ignored, deterministic prediction files containing record
ID, source, target, whitespace-token count, and a 12-decimal score—but no text.
Verification checks each file's bytes and SHA-256, then recomputes every metric
and subgroup table from those saved predictions.

## Overall results

All threshold metrics below use the fixed, unselected threshold 0.5. Target 1
means machine-generated.

| Model | Balanced accuracy | ROC AUC | Average precision | FPR | Recall | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.500000 | 0.500000 | 0.493001 | 1.000000 | 1.000000 | 0.291939 | 0.204910 |
| Length logistic | 0.506920 | 0.529177 | 0.531670 | 0.518549 | 0.532388 | 0.248731 | 0.019993 |
| Word TF-IDF logistic | 0.731238 | 0.814070 | 0.828215 | 0.266089 | 0.728565 | 0.176634 | 0.043285 |

The length-only diagnostic is close to chance despite the development length
tail difference. The lexical model has useful ranking discrimination, but the
default threshold falsely labels 6,814 of 25,608 human records as machine. Its
false-positive rate is 0.266089 with a Wilson 95% interval of
[0.260712, 0.271536]. It correctly labels 18,142 of 24,901 machine records and
misses 6,759.

These are validation results, not final test estimates. Because validation will
be used for calibration and threshold selection, it cannot also provide an
unbiased estimate of the resulting policy.

## Domain variation

The word TF-IDF model is not stable across domains at threshold 0.5.

| Domain | Records | Balanced accuracy | FPR | Recall | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 6,313 | 0.644531 | 0.525129 | 0.814191 | 0.750503 |
| Yelp | 5,221 | 0.667879 | 0.041776 | 0.377535 | 0.810150 |
| ROCStories | 6,447 | 0.669762 | 0.490196 | 0.829720 | 0.788499 |
| TL;DR | 4,956 | 0.719157 | 0.431287 | 0.869601 | 0.847130 |
| XSum | 6,529 | 0.720259 | 0.309713 | 0.750230 | 0.805730 |
| ELI5 | 6,309 | 0.736856 | 0.096949 | 0.570661 | 0.834417 |
| Scientific generation | 4,842 | 0.781607 | 0.207115 | 0.770329 | 0.864301 |
| SQuAD | 4,980 | 0.832553 | 0.082569 | 0.747675 | 0.918883 |
| CMV | 4,912 | 0.867121 | 0.099092 | 0.833333 | 0.950106 |

The 48.3-point range in human false-positive rates—from 4.1776% on Yelp to
52.5129% on HellaSwag—shows why one aggregate threshold cannot be described as
reliable across domains. Domain-specific thresholds would require operational
domain knowledge and separate validation; none are introduced here.

## Length and strategy slices

Word TF-IDF ranking is weakest below 50 whitespace tokens: ROC AUC is 0.570407,
balanced accuracy 0.551862, and FPR 0.438833 over 13,308 records. ROC AUC rises
to 0.787496 for 50–128 tokens, 0.931571 for 129–512, and 0.944979 above 512.
Length therefore remains an important limitation even though length alone is a
weak classifier.

Machine recall is 0.705711 for continuation, 0.857233 for specified, and
0.904159 for topical generation. These are dataset-specific strategy slices,
not evidence about all prompting behavior.

## Calibration and throughput

The word model's raw validation Brier score is 0.176634 and 15-bin equal-width
ECE is 0.043285. These are better than the controls but do not make the raw
logistic score a calibrated authorship probability. A separate validation-only
calibration and abstaining-threshold policy is required.

On the documented CPU, batch scoring all 50,509 validation texts took 15.148
seconds for word TF-IDF (3,334.271 records/second) and 0.726 seconds for the
length diagnostic. These are batch observations, not request-latency claims.
Model artifact sizes are documented separately in the training report.

## Conclusion

Word TF-IDF logistic regression becomes the baseline candidate for calibration
because it is the only model with materially useful validation ranking. It is
not selected as a final product model, and threshold 0.5 is explicitly
rejected for operational use. The high human false-positive rate, sharp domain
variation, and short-text failure justify an uncertain category and prominent
limitations in any later interface.
