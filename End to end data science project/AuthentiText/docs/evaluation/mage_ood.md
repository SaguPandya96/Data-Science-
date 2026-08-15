# Frozen baseline on MAGE development OOD stress sets

The frozen word TF-IDF model, isotonic calibrator, and two abstention thresholds
were applied without retuning to MAGE's two pinned development-only OOD files.
These files are not the sealed external test set and do not establish
cross-dataset generalization.

## Reproduce

```powershell
python scripts/evaluate_ood.py
python scripts/evaluate_ood.py --verify-only
```

The evaluator verifies both raw CSV hashes and both model artifact hashes,
writes deterministic text-free predictions, recomputes every reported metric
from those predictions, and confirms that no published test data is used.

The files repeat the same 762 human controls. Each file is therefore reported
separately. The combined view deduplicates exact content IDs and contains 3,162
unique texts: 762 human and 2,400 machine-generated or machine-paraphrased.

## Results

| Slice | Rows | ROC AUC | AP | Brier | ECE | Uncertain | Human → machine | Machine → human | Decisive accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4 OOD | 1,562 | 0.746868 | 0.753502 | 0.206653 | 0.065172 | 62.4200% | 12.3360% | 2.0000% | 81.2606% |
| Paraphrase OOD | 2,362 | 0.672621 | 0.791463 | 0.238462 | 0.185547 | 68.9670% | 12.3360% | 4.6250% | 77.0805% |
| Exact-content-deduplicated combined | 3,162 | 0.697370 | 0.867506 | 0.225546 | 0.218292 | 66.1607% | 12.3360% | 3.7500% | 82.8037% |

Average precision is prevalence-sensitive: the combined slice is 75.9%
machine-positive, so its higher AP must not be read as better generalization.
ROC AUC and the policy errors both show substantial degradation relative to the
sanitized in-distribution test.

The frozen three-way outcomes by unique-text source family are:

| Family | Rows | Likely human | Uncertain | Likely machine |
| --- | ---: | ---: | ---: | ---: |
| Human control | 762 | 20.4724% | 67.1916% | 12.3360% |
| GPT-4 | 800 | 2.0000% | 57.8750% | 40.1250% |
| GPT-4 paraphrase | 800 | 2.7500% | 67.2500% | 30.0000% |
| Machine-paraphrased human | 800 | 6.5000% | 72.3750% | 21.1250% |

The human-control false-machine rate varies sharply by domain: 32.7160% for
DialogSum, 18.5000% for CNN, 2.0000% for IMDb, and 0% for PubMed. These are
measured subgroup results, not evidence that any individual passage has a
particular origin.

## Interpretation and limits

The stress test confirms that abstention alone does not make the baseline
domain-robust. It also shows reduced decisive machine detection after
paraphrasing. No threshold or model change is permitted from these results;
they are evaluation evidence for later model selection.

There is no stable upstream pair identifier, so this report does not claim
paired before/after effects. Exact human duplicates are handled explicitly, but
near-duplicate or prompt-level dependence may remain. The four domains and
three machine families are small development slices from MAGE, not a population
sample. The prediction artifact contains IDs, labels, source metadata, lengths,
and scores, but no text.
