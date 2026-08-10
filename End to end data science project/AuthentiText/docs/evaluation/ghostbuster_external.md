# Frozen Ghostbuster external evaluation

The version 1 word TF-IDF model, isotonic calibrator, and two abstention
thresholds were frozen before Ghostbuster acquisition. After deterministic
preparation and the outcome-blind overlap gate, the policy was applied once to
the remaining 20,991 records. No result changed the model, calibration map,
thresholds, or exclusion rule.

## Reproduce and verify

The first command refuses to overwrite an existing report. The completed run
is therefore reproduced through verification-only mode:

```powershell
python scripts/evaluate_ghostbuster.py --verify-only
```

Verification checks the pinned repository and 20,994-row prepared-file
identity, the zero-cross-dataset-match overlap report, all three excluded
record IDs, the frozen model and calibrator hashes, and the validation-derived
thresholds. It recomputes every metric from a deterministic 1,173,100-byte
text-free prediction artifact and proves its IDs equal the prepared population
minus the exclusion set.

## Overall results

The scored data contain 2,992 human and 17,999 machine records, so machine
prevalence is 85.7463%. Average precision and accuracy are therefore not
directly comparable with the approximately balanced MAGE in-distribution test.

| Rows | ROC AUC | AP | Isotonic Brier | ECE | Uncertain | Human → machine | Machine → human | Decisive accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,991 | 0.828929 | 0.962187 | 0.130866 | 0.152084 | 42.9660% | 12.5334% | 1.3001% | 94.9131% |

The raw 0.5 score threshold has a 42.4131% human false-positive rate (1,269 of
2,992 human records), which reinforces that it is not the product decision
threshold. The frozen abstaining policy covers 57.0340% of rows: 602 are
`likely_human`, 9,019 are `uncertain`, and 11,370 are `likely_machine`. Among
decisive outcomes, 375 human records are called likely machine and 234 machine
records are called likely human. The Wilson 95% intervals are 11.3949%–13.7681%
and 1.1447%–1.4763%, respectively.

## Domain results

| Domain | Rows | Human | Machine | Uncertain | Human → machine | Machine → human | Decisive accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Creative writing | 6,999 | 1,000 | 5,999 | 44.2920% | 11.3000% | 1.0835% | 95.4347% |
| Reuters news | 6,998 | 998 | 6,000 | 38.4681% | 1.6032% | 0.3167% | 99.1872% |
| Student essays | 6,994 | 994 | 6,000 | 46.1395% | 24.7485% | 2.5000% | 89.4877% |

The student-essay human false-machine rate is the highest-cost finding: 246 of
994 usable human essays receive a likely-machine result. Its Wilson 95%
interval is 22.1663%–27.5251%. The external aggregate is therefore not evidence
that the detector is safe for student evaluation, even though its ranking and
machine-positive precision look favorable.

## Generator and prompt-strategy outcomes

These tables contain machine records only. “Likely machine” is sensitivity at
the frozen high threshold, not generator attribution.

| Generator | Rows | Likely human | Uncertain | Likely machine |
| --- | ---: | ---: | ---: | ---: |
| Claude | 3,000 | 6.1667% | 72.3333% | 21.5000% |
| GPT-3.5 Turbo | 14,999 | 0.3267% | 30.6687% | 69.0046% |

| Strategy | Rows | Likely human | Uncertain | Likely machine |
| --- | ---: | ---: | ---: | ---: |
| Original (ChatGPT plus Claude) | 6,000 | 3.2500% | 49.1833% | 47.5667% |
| Prompt 1 | 3,000 | 0.5333% | 39.3333% | 60.1333% |
| Prompt 2 | 2,999 | 0.3668% | 34.6782% | 64.9550% |
| Semantic | 3,000 | 0.1667% | 22.9000% | 76.9333% |
| Writing | 3,000 | 0.2333% | 30.4000% | 69.3667% |

Claude shift is pronounced: nearly three quarters of its records are
uncertain and only 21.5% reach likely machine. The strategy groups are not
independent generator comparisons because the original group combines two
generators and prompt variants can share source material.

## Interpretation and limits

This result supplies real cross-dataset evidence, but not production
validation. Ghostbuster is English-only, uses older ChatGPT and Claude outputs,
and has only three source domains. Its class prevalence is highly skewed, its
documents can share prompts and source material, and the bounded near-overlap
gate is lexical rather than semantic. Six oversized lexical blocks and one
tokenless MAGE record are explicitly outside the near-candidate census.

The external ECE of 15.2084% shows that the MAGE-derived probabilities do not
remain well calibrated under this shift. The policy also misses its 5% human
false-machine point goal by a wide margin, especially on essays. These
findings must accompany any external-performance claim. They do not authorize
post-hoc threshold changes; a new policy would require a separately identified
development and evaluation cycle with a new untouched final dataset.

The authoritative text-free report is
[`ghostbuster_evaluation_report.json`](../../data/metadata/ghostbuster_evaluation_report.json).
The [preparation and overlap protocol](../data/ghostbuster_main.md) documents
the source selection, six upstream blank files, three internal-copy exclusions,
and bounded overlap limitations.
