# Retraining and model-promotion design

## Scope and current status

AuthentiText has no automatic retraining job, model registry, promotion command,
or production deployment. The version 1 lexical artifact, isotonic calibrator,
thresholds, and published-test result are frozen. This document defines the
evidence and control flow required before a later candidate can replace them;
it does not claim that such a cycle has run.

Drift never retrains or promotes a model automatically. A drift flag is only a
reason to investigate whether a sufficiently representative, lawfully usable,
labeled dataset can be collected and validated.

## Valid and invalid triggers

A new candidate cycle may be proposed after:

- a credible public dataset or documented generator release adds a material,
  licensable evaluation or training population;
- repeated aggregate drift is confirmed with provenance-checked labeled data;
- a reproducible subgroup, robustness, calibration, privacy, or implementation
  failure is established; or
- a dependency or inference-contract change requires a new artifact.

A single alert, complaint, benchmark score, unlabeled request stream, test-set
failure, or desire for a newer model is not a sufficient trigger. Production
inputs must not be silently relabeled from the current model's predictions and
fed back into training.

## Cycle identity and immutable evidence

Every proposed cycle needs a unique identifier created when work begins. Its
records must include:

- Git commit, environment lock, Python/package versions, seed, and measured
  hardware;
- dataset names, immutable revisions, source URLs, licenses, retrieval times,
  file sizes, and SHA-256 digests;
- row-level stable identifiers and every filter, label mapping, removal reason,
  sampling rule, and split role;
- training configuration, pretrained-weight identity and license where
  applicable, artifact size/hash, convergence state, timing, and resource use;
- text-free predictions and recomputable evaluation, calibration, subgroup,
  robustness, and latency reports; and
- a signed or otherwise attributable manual gate decision with the prior model
  and rollback target.

Existing reports and artifacts are never overwritten. A failed or rejected run
keeps its real result and status. Raw text, large predictions, and model files
remain outside Git; their committed manifests and reports carry identities.

## Required sequence

### 1. Provenance intake

Inventory metadata before downloading large content. Confirm an official or
credible source, immutable version, access terms, redistribution limits,
language, labels, generators, domains, and expected size. Record unknown facts
as unknown. Paid data, APIs, compute, or storage require explicit permission.

### 2. Data validation

Verify file hashes and schema, then measure missing/blank text, invalid labels,
encoding, lengths, class/domain/generator balance, metadata completeness,
duplicate IDs, and exact duplicates. Reconcile every input row to a retained or
reason-coded removed row. A validation failure stops the cycle; it is not
converted into a warning merely to continue training.

### 3. Leakage and contamination analysis

Audit exact and normalized text equality, bounded near duplicates, source or
prompt relationships where identifiers exist, shared documents, templates,
metadata shortcuts, and overlap with all reserved evaluations. Conflicting
targets require an explicit policy. No record related to a sealed external role
may enter fitting, feature selection, calibration, threshold selection, or
error-driven iteration.

### 4. Versioned roles and split policy

Create immutable train, calibration-fit, policy-selection, calibration-audit,
development-OOD, and final-evaluation roles before model results are viewed.
Group related records, preserve source provenance, record seeds and algorithms,
and publish count/hash reconciliation. The MAGE published test used by version
1 remains historical evidence; repeatedly optimizing new candidates against it
would turn it into development data. A replacement needs a newly designated,
untouched final role or must state that no new final claim is available.

### 5. Candidate training

Start with the current baseline and the smallest justified challenger. Fit only
the training role. Record every model, feature, tokenizer, truncation, sampling,
weight, hyperparameter, seed, dependency, fit time, warning, and artifact hash.
Hyperparameter selection uses only predefined development roles. Failed and
poor runs remain in the experiment record.

### 6. Calibration and policy freeze

Fit candidate calibration on its calibration-fit role. Compare declared
methods on policy selection using Brier score and ECE, then select thresholds
under the predeclared cross-class error and coverage objectives. Audit the
complete policy once on the disjoint calibration-audit role. Freeze the model,
preprocessing, calibrator, thresholds, input rules, warnings, and aggregation
before any final or sealed evaluation.

### 7. Development OOD and robustness evaluation

Run predefined domain, generator, length, edit, and development-OOD conditions
without retuning from their outcomes. Report ROC AUC, prevalence-aware AP,
calibration, coverage, human false-machine, machine false-human, decisive
accuracy, intervals, subgroup counts, latency, memory, and artifact size.
Inspect high-cost errors using stable IDs and only license-conscious excerpts.

### 8. Sealed external acceptance

After one candidate is provisionally selected and frozen, run the newly
designated external role once. First check cross-dataset exact and bounded
near-overlap without moving external records into development. Save immutable,
text-free predictions and report every planned slice. If the candidate fails,
do not adjust it against the external set; begin a new cycle with new evidence
and a new valid final-evaluation plan.

### 9. Candidate comparison and acceptance gates

The gate definition is versioned before fitting. It must include:

- complete provenance, validation, leakage, split, and artifact-integrity
  checks with no unresolved failure;
- both validation-audit cross-class decisive-error point targets, including the
  existing 5% human false-machine and 5% machine false-human research targets,
  plus a minimum useful coverage chosen before scoring;
- no predeclared material regression in calibration, domain, generator,
  short-text, robustness, external, latency, memory, or package-size evidence;
- a meaningful predeclared improvement over the retained baseline on at least
  one high-cost safety or generalization outcome, not accuracy or AUC alone;
- intervals and sample counts adequate for the claim, with inconclusive results
  treated as inconclusive rather than passes;
- a reloaded artifact whose predictions and reports reproduce from fixed
  inputs; and
- updated model card, responsible-AI guidance, inference/API contracts,
  privacy review, and tests.

Numeric coverage and materiality margins depend on the intended use and new
evaluation population, so they must be fixed in that cycle's protocol. They
cannot be selected after viewing candidate results. Passing these gates can
justify replacement of a research baseline; it does not by itself establish
production or high-stakes suitability.

### 10. Manual promotion

Promotion requires a reviewed record naming every gate as pass, fail, or
inconclusive; linking the candidate and prior artifact hashes; identifying the
decision maker and time; and naming the exact rollback version. The service
must then pass package, startup, readiness, health, prediction, batch, privacy,
monitoring, and drift-identity acceptance checks with the promoted artifacts.

There is no approved promotion mechanism in version 1. Replacing local files by
hand, reusing an artifact filename without a new hash-linked report, or changing
thresholds through configuration is not a promotion record.

### 11. Rollback and post-promotion observation

Keep the prior verified artifacts, dependency lock, reports, and service
configuration available. Roll back on artifact/readiness failure, contract
breakage, privacy leakage, sustained service errors, or a reviewed safety
incident. A rollback restores the prior complete model/calibrator/threshold
set; it never mixes components across versions and never erases the rejected
evidence.

Post-promotion monitoring begins empty and observes actual aggregate traffic.
It cannot estimate model accuracy without independent, provenance-checked
labels. Drift and error signals trigger review, not threshold changes or a
background training job.

## Required promotion record

A real promotion record is created only when a candidate exists. It must bind:

- candidate cycle ID, Git commit, environment, dataset manifests, and report
  hashes;
- base model, calibration artifact, thresholds, schema, and package hashes;
- gate definitions fixed before training and their measured outcomes;
- known failures, waivers, responsible-AI review, and external-evaluation
  status;
- approver identity, decision time, deployment target, and rollback identity;
  and
- acceptance-test outputs and the observation start time.

Missing, failed, or inconclusive gates remain visible. No field may be inferred
from an artifact filename or replaced with an unverified narrative claim.

## Unimplemented dependencies

Version 1 has no persistent feedback labels, model registry, experiment server,
automated candidate pipeline, container build, deployment target, secret store,
traffic router, or rollback command. The original Ghostbuster final evaluation
is complete and immutable, but no new untouched final evaluation is designated
for a future candidate cycle. Therefore this design closes the retraining
policy requirement only; it does not complete model retraining, Docker, or
deployment phases.
