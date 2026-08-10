# MAGE leave-one-exact-generator-out evaluation

## Question and protocol

This experiment measures transfer to one exact upstream generator identifier
whose machine rows are absent from model fitting, calibration fitting,
calibration-method selection, threshold selection, and calibration audit.

For each of the 27 declared regimes, the runner:

1. keeps every human row but excludes machine rows from the held generator in
   published train and validation;
2. fits majority, length-only, and word TF-IDF logistic controls;
3. independently selects an isotonic calibrator and two abstention thresholds
   from three disjoint validation roles;
4. only then materializes published-test human rows and held-generator machine
   rows;
5. scores and verifies the frozen fold; and
6. checkpoints text-free evidence before advancing.

The held-generator test role is intentionally imbalanced because it retains all
25,634 human negatives for false-positive analysis. Machine prevalence ranges
from 2.3243% to 7.1165% across folds, so average precision and Brier score are
not directly comparable with the balanced in-distribution test.

## Results

All values below come from the committed
[`mage_generator_holdout_report.json`](../../data/metadata/mage_generator_holdout_report.json).
Every fold selected its own isotonic calibrator and thresholds before reading
its test role.

| Held generator | Test rows | ROC AUC | AP | Brier | ECE | Uncertain | Human false-machine | Machine false-human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `13B` | 26,458 | 0.782478 | 0.224129 | 0.154393 | 0.310730 | 68.1420% | 4.4472% | 6.0680% |
| `30B` | 26,473 | 0.759302 | 0.184229 | 0.155315 | 0.310000 | 65.0852% | 5.0012% | 8.4625% |
| `65B` | 26,468 | 0.750378 | 0.170351 | 0.154947 | 0.308849 | 61.6783% | 4.6735% | 9.4724% |
| `7B` | 26,462 | 0.779783 | 0.226884 | 0.155801 | 0.312242 | 67.0433% | 6.7996% | 4.1063% |
| `GLM130B` | 26,454 | 0.803724 | 0.251777 | 0.154089 | 0.313004 | 57.7191% | 4.7710% | 7.0732% |
| `bloom_7b` | 26,403 | 0.906497 | 0.497305 | 0.153765 | 0.322423 | 65.1176% | 4.8763% | 1.3004% |
| `flan_t5_base` | 26,477 | 0.668022 | 0.066350 | 0.155046 | 0.302163 | 59.1608% | 4.9349% | 16.0142% |
| `flan_t5_large` | 26,473 | 0.649648 | 0.059960 | 0.154554 | 0.299769 | 66.1844% | 4.7632% | 13.7068% |
| `flan_t5_small` | 26,465 | 0.651646 | 0.058858 | 0.154491 | 0.300984 | 57.7215% | 5.3991% | 18.7726% |
| `flan_t5_xl` | 26,448 | 0.636999 | 0.055240 | 0.155145 | 0.300394 | 57.9477% | 6.2417% | 18.6732% |
| `flan_t5_xxl` | 26,469 | 0.657153 | 0.057654 | 0.154421 | 0.300573 | 68.3857% | 4.9583% | 10.4192% |
| `gpt-3.5-trubo` | 27,598 | 0.842777 | 0.396462 | 0.151988 | 0.283795 | 67.0809% | 6.5733% | 1.3747% |
| `gpt_j` | 26,317 | 0.873035 | 0.372625 | 0.155080 | 0.322028 | 65.1100% | 5.1611% | 2.6354% |
| `gpt_neox` | 26,244 | 0.837473 | 0.276982 | 0.154857 | 0.322199 | 64.3652% | 6.8464% | 3.2787% |
| `opt_1.3b` | 26,450 | 0.845278 | 0.371712 | 0.154304 | 0.316586 | 65.2023% | 5.4732% | 3.7990% |
| `opt_125m` | 26,434 | 0.854627 | 0.375661 | 0.153774 | 0.317418 | 62.9076% | 4.9739% | 3.3750% |
| `opt_13b` | 26,357 | 0.842286 | 0.341216 | 0.154081 | 0.318576 | 66.7527% | 5.1338% | 3.4578% |
| `opt_2.7b` | 26,438 | 0.851447 | 0.395758 | 0.153472 | 0.316969 | 58.2949% | 3.8894% | 5.8458% |
| `opt_30b` | 26,446 | 0.851909 | 0.376574 | 0.153970 | 0.317072 | 61.7522% | 7.3613% | 3.5714% |
| `opt_350m` | 26,428 | 0.900439 | 0.576487 | 0.154014 | 0.322583 | 54.4347% | 6.1832% | 4.6599% |
| `opt_6.7b` | 26,438 | 0.833692 | 0.287059 | 0.154491 | 0.315882 | 64.2371% | 5.3601% | 4.3532% |
| `opt_iml_30b` | 26,447 | 0.816044 | 0.260767 | 0.154061 | 0.313189 | 58.2221% | 4.8061% | 5.7811% |
| `opt_iml_max_1.3b` | 26,465 | 0.803100 | 0.243773 | 0.154283 | 0.311882 | 65.5583% | 5.1728% | 3.9711% |
| `t0_11b` | 26,414 | 0.697761 | 0.073209 | 0.153542 | 0.304980 | 65.1094% | 4.5955% | 8.2051% |
| `t0_3b` | 26,476 | 0.688901 | 0.079806 | 0.154457 | 0.303129 | 69.9010% | 5.2469% | 7.9572% |
| `text-davinci-002` | 27,507 | 0.795519 | 0.305653 | 0.150343 | 0.275423 | 64.7690% | 6.1949% | 4.4314% |
| `text-davinci-003` | 27,542 | 0.830521 | 0.401984 | 0.150962 | 0.281926 | 65.0098% | 6.3197% | 2.5157% |

| Across 27 folds | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.636999 | 0.803724 | 0.906497 |
| Average precision | 0.055240 | 0.260767 | 0.576487 |
| Brier score | 0.150343 | 0.154304 | 0.155801 |
| Expected calibration error | 0.275423 | 0.311882 | 0.322583 |
| Coverage | 30.0990% | 34.9902% | 45.5653% |
| Uncertain rate | 54.4347% | 65.0098% | 69.9010% |
| Human false-machine rate | 3.8894% | 5.1611% | 7.3613% |
| Machine false-human rate | 1.3004% | 4.6599% | 18.7726% |

## Interpretation

Transfer varies strongly by exact generator. Fourteen of 27 ROC AUC values are
below the 0.806435 frozen in-distribution aggregate. The five Flan-T5 folds all
exceed a 10% machine false-human rate; `flan_t5_small` reaches 18.7726% and
`flan_t5_xl` has the minimum ROC AUC, 0.636999.

Abstention remains high, with median uncertainty of 65.0098%. Calibration does
not transfer: every fold has ECE above 27%, despite low Brier scores driven in
part by the low machine prevalence. Sixteen folds exceed a 5% human
false-machine rate, while the maximum reaches 7.3613% on `opt_30b`.

These are exact-identifier holdouts, not generator-family or external tests.
Related model sizes and architectures can remain in development when one exact
identifier is held out, and all folds share MAGE construction conventions.
The results therefore document source dependence without establishing
cross-dataset generalization.

## Reproduce or verify

With the ignored sanitized MAGE split present:

```powershell
python scripts/run_generator_holdouts.py
python scripts/run_generator_holdouts.py --resume
python scripts/run_generator_holdouts.py --verify-only
```

The complete run produced 216 ignored model, calibrator, and text-free
prediction files totaling 269.41 MiB. The separate verify-only pass reloads
every artifact, validates linkage and identity, and recomputes all stored
validation and test metrics. No selected fold text remains.
