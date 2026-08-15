# Experiment log

## Tracking approach

AuthentiText uses one deterministic, text-free
[`experiment_registry.json`](../data/metadata/experiment_registry.json) as the
index to completed local runs. Each entry is generated from a validated report,
records that report's SHA-256 digest, names the milestone commit, retains the
measured evidence needed for the decision, and marks its real completion
status. The source reports remain the authoritative detailed records.

MLflow or another tracking service is not justified for this single-machine,
single-cycle baseline. The committed reports already capture fixed inputs,
configuration, environment, artifact and prediction identities, metrics,
timing, and validation state without a service or database. A registry service
can be reconsidered if concurrent candidate runs, remote artifact storage, or
multiple approvers create a concrete coordination need.

## Completed runs

| Experiment ID | Role | Milestone | Authoritative report | Recorded decision |
| --- | --- | --- | --- | --- |
| `baseline_training_v1` | Fit three CPU controls on sanitized train only | `23fe60c` | [`mage_baseline_training_report.json`](../data/metadata/mage_baseline_training_report.json) | Retain all three fitted controls for validation |
| `baseline_validation_v1` | Compare controls on sanitized validation | `5782059` | [`mage_baseline_validation_report.json`](../data/metadata/mage_baseline_validation_report.json) | Advance word TF-IDF; reject raw threshold 0.5 for decisions |
| `calibration_policy_v1` | Select calibration and abstention on disjoint validation roles | `614c59c` | [`mage_calibration_report.json`](../data/metadata/mage_calibration_report.json) | Select isotonic calibration and freeze the three-way policy |
| `frozen_id_test_v1` | Score the sanitized published test once | `668d870` | [`mage_frozen_test_report.json`](../data/metadata/mage_frozen_test_report.json) | Preserve both missed error targets without retuning |
| `validation_drift_backtest_v1` | Select and audit aggregate drift thresholds on validation only | `8fb484d` | [`mage_drift_reference.json`](../data/metadata/mage_drift_reference.json) | Use every flag for investigation only |
| `mage_development_ood_v1` | Apply the frozen policy to GPT-4/paraphrase development stress sets | `c8dcec2` | [`mage_ood_evaluation_report.json`](../data/metadata/mage_ood_evaluation_report.json) | Retain the degraded OOD result without retuning |
| `mage_domain_holdout_v1` | Refit and recalibrate with each MAGE domain excluded in turn | `49c7fd8` | [`mage_domain_holdout_report.json`](../data/metadata/mage_domain_holdout_report.json) | Retain measured domain dependence without test-driven fold retuning |
| `mage_generator_holdout_v1` | Refit and recalibrate with each exact MAGE generator excluded in turn | `1f5ae54` | [`mage_generator_holdout_report.json`](../data/metadata/mage_generator_holdout_report.json) | Retain exact-generator dependence without family-level claims |
| `ghostbuster_external_evaluation` | Apply the unchanged frozen policy once to the overlap-gated external corpus | `deae0f0` | [`ghostbuster_evaluation_report.json`](../data/metadata/ghostbuster_evaluation_report.json) | Retain external calibration and human false-machine failures without retuning |
| `mage_truncation_robustness_v1` | Compare complete eligible MAGE test records with prespecified 50-, 100-, and 200-token prefixes | `54649bd` | [`mage_truncation_robustness_report.json`](../data/metadata/mage_truncation_robustness_report.json) | Retain measured prefix sensitivity without model, calibration, or threshold retuning |

The structured registry contains the actual row counts, ranking, calibration,
policy, throughput, artifact, false-alert, and shift-detection values extracted
from those reports. The [model card](MODEL_CARD.md) presents the key model and
safety evidence and is checked separately against the same sources.

## Explicitly not run

The registry also records these planned or considered experiments as unrun:

| Experiment ID | Status and reason |
| --- | --- |
| `transformer_candidate` | [Train-only preflight](../data/metadata/transformer_preflight_report.json) pins BERT-Tiny and verifies local resources; missing framework packages and weights block training, so no transformer was trained or evaluated |
| `raid_robustness_evaluation` | No storage-safe source-group acquisition plan has run |
| `multilingual_evaluation` | The first research cycle is English-only |

An unrun entry contains no metric, inferred outcome, or synthetic placeholder.
It prevents a planned experiment from being confused with evidence.

## Reproduction and integrity

Rebuild the registry from the committed reports, or verify the committed copy:

```powershell
python scripts/build_experiment_registry.py
python scripts/build_experiment_registry.py --check
```

The build is deterministic and has no data/model dependency beyond the ten
committed reports. The checker fails if a source report, extracted value,
source-report hash, decision, or registry formatting changes. CI runs the
read-only check.

Reproducing a source experiment is a stronger operation. It requires the
ignored data, model, and prediction artifacts named and hashed by that report;
the relevant script's `--verify-only` mode recomputes its validation evidence.
The registry does not replace those artifacts and does not turn stored metrics
into a fresh experiment.

## New experiment rule

A new entry is added only after its code actually ran, its report passed
validation, its artifacts or predictions were hash-linked, the diff was
reviewed, and the coherent milestone was committed. Failed experiments should
be retained with their real status when a report format for failures exists;
they must not be omitted to make the history look stronger. Candidate
replacement follows the [retraining and promotion design](operations/retraining.md).
