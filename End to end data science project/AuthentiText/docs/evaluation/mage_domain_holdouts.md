# MAGE leave-one-domain-out evaluation

## Question and protocol

This experiment measures whether the selected lexical baseline transfers to a
MAGE domain that is absent from model fitting, calibration fitting, calibration
method selection, threshold selection, and calibration audit.

For each of the nine declared regimes, the runner:

1. selects published train and validation rows outside the held domain;
2. fits majority, length-only, and word TF-IDF logistic controls;
3. scores the domain-excluded validation role;
4. independently selects calibration and two abstention thresholds from the
   same three disjoint validation roles used by the first cycle;
5. only then materializes and scores published-test rows from the held domain;
   and
6. reloads artifacts, verifies hashes/linkage, and recomputes prediction
   metrics.

Text is never a model metadata feature. Temporary selected text is removed
after every fold. The retained local artifacts are ignored by Git and include
three models, one calibrator, three text-free validation-prediction files, and
one text-free test-prediction file per fold.

## Results

All values below come from the committed
[`mage_domain_holdout_report.json`](../../data/metadata/mage_domain_holdout_report.json).
Each fold uses its own validation-selected isotonic calibrator and thresholds.

| Held domain | Test rows | ROC AUC | AP | Brier | ECE | Uncertain | Human false-machine | Machine false-human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cmv` | 4,909 | 0.790932 | 0.813803 | 0.188402 | 0.044945 | 67.9364% | 3.5833% | 4.0255% |
| `eli5` | 6,351 | 0.702483 | 0.731749 | 0.220005 | 0.055694 | 75.8621% | 6.5589% | 3.4429% |
| `hswag` | 6,346 | 0.642912 | 0.650331 | 0.252569 | 0.125160 | 67.5859% | 19.6841% | 2.0301% |
| `roct` | 6,456 | 0.678623 | 0.710152 | 0.232323 | 0.096889 | 71.2515% | 13.4251% | 1.8519% |
| `sci_gen` | 4,788 | 0.700217 | 0.721720 | 0.216998 | 0.065585 | 74.6867% | 7.4074% | 3.0667% |
| `squad` | 5,004 | 0.718838 | 0.731836 | 0.216514 | 0.046244 | 71.6427% | 10.8453% | 1.5224% |
| `tldr` | 4,977 | 0.765186 | 0.788054 | 0.215002 | 0.136629 | 61.8244% | 16.4892% | 0.6552% |
| `xsum` | 6,537 | 0.625869 | 0.652767 | 0.254679 | 0.129214 | 63.9131% | 21.3220% | 4.1487% |
| `yelp` | 5,199 | 0.705193 | 0.716796 | 0.231964 | 0.110378 | 57.1841% | 23.0015% | 1.8453% |

| Across nine folds | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.625869 | 0.702483 | 0.790932 |
| Average precision | 0.650331 | 0.721720 | 0.813803 |
| Brier score | 0.188402 | 0.220005 | 0.254679 |
| Expected calibration error | 0.044945 | 0.096889 | 0.136629 |
| Coverage | 24.1379% | 32.0636% | 42.8159% |
| Uncertain rate | 57.1841% | 67.9364% | 75.8621% |
| Human false-machine rate | 3.5833% | 13.4251% | 23.0015% |
| Machine false-human rate | 0.6552% | 2.0301% | 4.1487% |

## Interpretation

The experiment confirms substantial domain dependence. Every held-domain ROC
AUC is below the 0.806435 frozen in-distribution aggregate, with `xsum` at
0.625869 and `hswag` at 0.642912. Median coverage is only 32.0636%, so the
independently calibrated policies abstain on most held-domain examples.

Abstention does not solve the principal safety failure. Five of nine folds
exceed a 10% human false-machine rate, and `yelp` reaches 23.0015%. By contrast,
machine false-human rates stay below 5% in every fold. This asymmetry makes the
baseline especially unsuitable for punitive or authorship-consequential use.

These are within-dataset domain-holdout results, not external validation.
Related source conventions, generator families, benchmark construction, and
lexical artifacts can still appear on both sides of a fold. Ghostbuster remains
the sealed cross-dataset evaluation, and the 27 exact-generator folds remain
unrun.

## Reproduce or verify

With the ignored sanitized MAGE split present:

```powershell
python scripts/run_domain_holdouts.py
python scripts/run_domain_holdouts.py --verify-only
```

The complete run fitted nine independent model/calibration cycles and produced
72 ignored files totaling 73.95 MiB. The verify-only pass reloads every model
and calibrator, validates artifact identities and linkage, and recomputes all
stored validation and test metrics from text-free predictions.
