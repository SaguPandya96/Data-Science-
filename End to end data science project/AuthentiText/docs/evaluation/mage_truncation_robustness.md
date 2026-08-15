# MAGE prefix-truncation robustness protocol

This experiment measures how the frozen AuthentiText word TF-IDF policy changes
when an otherwise unchanged sanitized MAGE test document is shortened to a
prefix. It is a post-freeze stress test, not another model-selection or
threshold-selection stage.

## Prespecified conditions

Before evaluating outcomes, the protocol fixes three budgets: 50, 100, and 200
whitespace-delimited tokens. A condition includes only records whose original
length is strictly greater than its budget. The comparison therefore uses the
same records on both sides: the complete original and its truncated prefix.

The transformation ends the string at the final character of the budget's
last non-whitespace token. Every byte before that cut is preserved. Nothing is
inserted, normalized, paraphrased, or reordered. These budgets align with the
existing short-text warning boundary and provide two larger, round-number
prefix sizes; they are not selected from test outcomes.

The frozen base model, isotonic calibrator, likely-human threshold, and
likely-machine threshold remain unchanged. Results may document limitations
but may not change any of those artifacts or thresholds.

## Evidence contract

For each budget, the report records original and truncated ranking,
calibration, and three-way policy metrics on the paired eligible subset. It
also records raw and calibrated score deltas, category-change rates by target,
and the complete three-by-three category transition table. Deltas are always
defined as truncated minus original.

The ignored deterministic prediction artifact contains stable record IDs,
targets, source metadata, token counts, scores, categories, and budgets for
verification. It contains neither original nor transformed text. The committed
aggregate report likewise contains no source text.

## Run and verify

After the pinned sanitized split, model, calibrator, and validation prediction
artifacts are present, run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_truncation_robustness.py
.\.venv\Scripts\python.exe scripts\evaluate_truncation_robustness.py --verify-only
```

The first command writes the ignored paired prediction file and the committed
report target at
`data/metadata/mage_truncation_robustness_report.json`. The second command
checks prediction size and SHA-256, recalibrates every stored raw score,
reapplies the frozen thresholds, and recomputes every reported condition.

## Measured results

The prespecified run completed and verified 77,080 paired rows. Because a
record must be longer than its condition's budget, the eligible populations
shrink as the prefix grows. Every comparison within a row remains paired.

| Budget | Paired rows | Original ROC AUC | Prefix ROC AUC | Original uncertain | Prefix uncertain | Human â†’ machine, original | Human â†’ machine, prefix | Machine â†’ human, original | Machine â†’ human, prefix | Category changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 36,965 | 0.878027 | 0.671298 | 50.8346% | 69.7849% | 3.3848% | 6.7473% | 4.1176% | 9.5497% | 36.0801% |
| 100 | 25,887 | 0.923264 | 0.805082 | 45.3896% | 62.7187% | 2.3518% | 3.5514% | 2.5575% | 5.9625% | 30.0846% |
| 200 | 14,228 | 0.945366 | 0.886284 | 39.5558% | 52.4670% | 2.1611% | 2.1611% | 1.7895% | 3.9445% | 23.1094% |

At 50 tokens, ROC AUC falls by 0.206729 and uncertainty rises by 18.9503
percentage points. Mean raw scores move upward by 0.101876 for human records
and downward by 0.161603 for machine records, compressing the two classes
toward and across the uncertain region. Abstention absorbs many changes but
does not prevent decisive errors: both cross-class error rates more than
double in the 50-token condition.

The effect weakens with longer prefixes but remains material at 200 tokens,
where 23.1094% of categories change and machine false-human rises from 1.7895%
to 3.9445% on the eligible paired subset. No result changed the frozen model,
calibrator, thresholds, or product inference behavior.

## Scope and limitations

This intervention measures prefix removal, not naturally short documents,
middle deletion, summaries, paraphrases, mixed authorship, or an adversary's
adaptive edit. Conditions overlap and are not independent samples. The
eligible original controls also differ across budgets, so results should be
compared within a budget rather than treating the three original columns as
one population trend.

The aggregate source report is
[`mage_truncation_robustness_report.json`](../../data/metadata/mage_truncation_robustness_report.json).
Its paired prediction artifact is ignored by Git and can be independently
verified only when the pinned split and frozen artifacts are available.
