# Dataset comparison and initial evaluation design

Research date: 2026-08-08.

This review uses author repositories, dataset cards, and primary papers. A
repository-level license is not assumed to resolve every upstream human-text
license. Facts that could not be verified are recorded as `not reported` or
`unknown` in the registry rather than inferred.

The machine-written class is a dataset label, not proof about authorship. The
project will preserve the source label while using the user-facing language
`likely human-written`, `uncertain`, and `likely machine-generated` only after
calibration and threshold validation.

## Shortlist

| Dataset | Evidence relevant to this project | Initial role | Decision |
| --- | --- | --- | --- |
| MAGE | English, ten core sources, 27 core LLMs, source metadata, published splits, and separate GPT-4/paraphrase sets | Development corpus | Select |
| Ghostbuster | Human essays, Reuters news, creative writing, multiple prompt strategies, ChatGPT and Claude, explicit CC BY 3.0 data license | Sealed final external evaluation | Select |
| RAID | Eleven model variants, eleven domains, decoding variation, source IDs, and adversarial variants | External robustness audit | Conditional on a bounded acquisition plan |
| HC3 English | Five QA sources and paired human/ChatGPT answers | Legacy diagnostic | Do not select initially |
| M4 | Seven languages, multiple domains, six generators, and paired examples | Multilingual candidate | Defer |
| SemEval-2024 Task 8 | M4 extension with surprise domains, generators, languages, and a mixed-authorship task | Shared-task comparison | Defer |
| DetectRL | Realistic prompt, revision, noise, mixing, and length scenarios | Robustness candidate | Defer until licensing and release packaging are clear |
| DetectRL-X | Current multilingual, refinement, and attack design across eight languages | Future external candidate | Watchlist; data was not public at inspection |

The machine-readable details, including record-count caveats and access notes,
are in [`data/metadata/dataset_registry.csv`](../../data/metadata/dataset_registry.csv).

## Primary-source findings

### MAGE

The [ACL 2024 paper](https://aclanthology.org/2024.acl-long.3/) and
[official repository](https://github.com/yafuly/MAGE) describe a corpus of
human and machine texts from ten core sources and 27 core LLMs, plus two small
OOD sets for a new model/domain combination and sentence-level paraphrasing.
The [Hugging Face release](https://huggingface.co/datasets/yaful/MAGE/tree/342663f)
lists about 554 MB of CSV data and an Apache-2.0 tag.

The repository reports 447,674 texts, while the human/machine counts in Table 6
of the paper sum to 448,459. This is not silently resolved: acquisition must pin
the revision, count the actual rows by split and label, and document the cause
if it can be established. The repository license also does not eliminate the
need to record the terms of the ten upstream human datasets.

MAGE is the strongest practical development source for the inspected machine:
it is materially more diverse than HC3, far smaller than RAID, and supplies
the source field needed for domain and generator analysis. Its published random
split is provisional until prompt/source duplicate leakage is measured.

### Ghostbuster

The [NAACL 2024 paper](https://aclanthology.org/2024.naacl-long.95/) reports
three 7,000-document domains: 1,000 human, 5,000 ChatGPT, and 1,000 Claude
documents per domain. The [official data repository](https://github.com/vivek3141/ghostbuster-data)
uses CC BY 3.0 and includes post-November-2023 data, prompt variants, and
perturbation/auxiliary evaluation folders.

This corpus is small enough to seal as a complete external evaluation and is
directly relevant to the costliest error: legitimate student writing flagged as
machine-generated. It is not perfectly independent of MAGE. Both use material
from Reddit WritingPrompts, so MAGE's WritingPrompts (`WP`) source will be
excluded from all model fitting, feature selection, calibration, and threshold
selection. Exact and near-duplicate checks across remaining sources are still
required before the external run.

The external corpus may be acquired and hashed, but its text will not be
sampled, explored, or scored before the model, calibration method, and decision
thresholds are frozen. Dataset-level counts, paths, schemas, checksums, and
license files may be inspected without exposing example content.

### RAID

The [ACL 2024 paper](https://aclanthology.org/2024.acl-long.674/),
[official repository](https://github.com/liamdugan/raid), and
[revision-pinned dataset card](https://huggingface.co/datasets/liamdugan/raid/tree/865cac7)
describe source IDs, eleven model variants, eleven domains, four decoding and
repetition configurations, and eleven named attacks plus the clean condition.
The current Hugging Face release is 16.7 GB. The repository says it contains
over ten million documents, while the dataset viewer currently summarizes
about 7.42 million rows. Actual acquired counts must therefore be measured.

The complete release is larger than the free space on the current C: drive.
RAID remains selected only conditionally: a later design must select
source-disjoint records deterministically without downloading or materializing
the entire adversarial corpus. The public RAID test labels are withheld, so any
local labeled audit must use a clearly identified, untouched subset of the
labeled training release and must not be presented as the official leaderboard
test.

### Other credible candidates

- [HC3](https://arxiv.org/abs/2301.07597) has 24,322 English question rows
  containing 58,546 human and 26,903 ChatGPT answer strings. The
  [official repository](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
  says CC BY-SA applies only when an upstream source does not impose stricter
  terms and identifies unknown terms for the medicine and finance sources. A
  single early generator and nested same-question answers make it weaker than
  MAGE for initial development.
- [M4](https://aclanthology.org/2024.eacl-long.83/) contains 147,895 parallel
  texts across seven languages and six generators. Its paper documents
  source-specific licenses, including non-commercial terms; the
  [official repository](https://github.com/mbzuai-nlp/M4) has no release-level
  license or tagged version. It remains a strong future multilingual candidate.
- [SemEval-2024 Task 8](https://aclanthology.org/2024.semeval-1.279/) extends M4
  with unseen domains, generators, languages, and a mixed-authorship boundary
  task. Its [official repository](https://github.com/mbzuai-nlp/SemEval2024-task8)
  is Apache-2.0, but the task data overlaps M4 and its public test labels invite
  accidental repeated tuning. It is deferred rather than mixed into the first
  training pool.
- [DetectRL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b61bdf7e9f64c04ec75a26e781e2ad51-Abstract-Datasets_and_Benchmarks_Track.html)
  is scientifically attractive for prompt, revision, spelling, mixing, and
  length stress tests. The [official repository](https://github.com/NLP2CT/DetectRL)
  did not state a license, packaged record count, or immutable direct release in
  the inspected landing page, so it is deferred.
- [DetectRL-X](https://aclanthology.org/2026.acl-long.1773/) is the most current
  reviewed candidate found, covering eight languages, six domains, four
  commercial LLMs, refinement operations, paraphrases, and perturbations. Its
  authors' Hugging Face organization had no public dataset at inspection, so no
  license, row count, or access claim is made.

## Initial evaluation protocol

The first research cycle is English-only. Multilingual performance is a
separate question requiring multilingual data, language-aware analysis, and a
suitable encoder; it will not be implied by English results.

### Development data

MAGE has the following development roles after the documented leakage controls:

1. Official train split, excluding `WP`, is the candidate fitting population.
2. Official validation split, excluding `WP`, is used for model selection,
   calibration fitting, and empirical decision-threshold selection.
3. Official test split, excluding `WP`, is the held-out in-distribution test and
   is not used for model or threshold choices.
4. The two MAGE OOD splits are development-time stress tests. They may reveal
   weaknesses, but their results cannot be used to tune the final external
   decision threshold.

The published partitions are sanitized by normalized equality and confirmed
sampled lexical-overlap relationships. MAGE does not expose a stable prompt ID
in the acquired CSV schema; its `src` value does encode domain, human/machine
kind, generation strategy, and generator. The resulting selector-only
source-holdout regimes are documented in
[the source-regime record](mage_source_regimes.md). Cross-dataset overlap with
Ghostbuster was later checked through population exact and normalized equality
plus bounded population lexical blocking; no confirmed cross-dataset match was
found. No Ghostbuster example was moved into development partitions.

Any compute-bounded sample will be selected by a recorded hash of stable record
identifiers within label, domain, and generator strata. A row-order prefix is
not an acceptable sample. The seed, algorithm, population counts, selected
counts, and exclusions must be saved.

### Generalization regimes

- **In-distribution:** MAGE held-out test after leakage controls.
- **Domain holdout:** rotate supported MAGE domains, never moving related source
  or prompt groups across the boundary.
- **Generator holdout:** hold out a generator family supported by the acquired
  metadata; the exact family is chosen before model results are viewed.
- **Development robustness:** MAGE unseen-domain/unseen-GPT-4 and paraphrase
  sets, reported separately.
- **Final external:** the sealed Ghostbuster corpus, with results broken out for
  essays, news, creative writing, generator, prompt strategy, and human false
  positives.
- **Conditional external robustness:** a future source-disjoint RAID sample,
  reported separately from its official hidden-label test.

### Metrics and uncertainty

The machine label is the positive class for metric definitions. Reports will
include accuracy, precision, recall, F1, macro F1, AUROC, AUPRC, confusion
matrix, false-positive rate, false-negative rate, Brier score, ECE, latency,
and model size where applicable. Human false-positive rate is explicitly:

> human-written records classified as likely machine-generated divided by all
> evaluated human-written records

Binary discrimination metrics will be reported independently from the
three-way abstaining decision. Thresholds will be chosen only on validation
data under a documented human false-positive constraint; no numeric constraint
is selected before validation distributions exist. Confidence intervals will
use grouped resampling by source/prompt where dependence is known, rather than
pretending every generated variant is independent.

Breakdowns will cover dataset, domain, generator, prompt family, text-length
band, and attack condition when those fields are available. No short-text
minimum, long-text chunk size, calibration method, or abstention boundary is
fixed until measured evidence supports it.

### External-test governance

The final Ghostbuster evaluation happened only after the model artifact,
preprocessing version, feature vocabulary, calibration fit, short/long text
policy, and thresholds are frozen and recorded. The evaluation writes immutable
predictions and a run manifest. If an implementation fault invalidates a run,
the fault, affected outputs, and reason for any rerun must be documented. Weak
external results remain part of the record and cannot trigger tuning against
the external examples. The completed
[external report](../evaluation/ghostbuster_external.md) preserves the weak
human false-machine and calibration outcomes.

## Compute and acquisition implications

The workstation has 15.82 GiB RAM, no CUDA-capable GPU tooling, and limited
local disk headroom. The practical sequence is therefore:

1. Acquire revision metadata and small MAGE OOD files before bulk data.
2. Stream and count the chosen MAGE revision; do not assume paper counts.
3. Profile a deterministic development slice before setting a full baseline
   sample ceiling or TF-IDF vocabulary size.
4. Run sparse linear baselines locally and record measured memory, fit time,
   inference time, and artifact size.
5. Benchmark a small transformer pilot before choosing between CPU training and
   a separately approved free GPU environment.
6. Do not download full RAID unless a bounded, reproducible plan fits disk and
   transfer constraints.

No paid API, hosted GPU, database, or cloud resource is required for the
current research and acquisition stages.

## Open checks before modeling

- Reconcile MAGE release counts with the paper and record per-split checksums.
- Resolve or enforce the upstream licenses for selected MAGE sources.
- Keep the completed pinned Ghostbuster acquisition, overlap audit, and frozen
  evaluation immutable; use verification-only mode for reproduction.
- Determine whether a stable, source-disjoint RAID sample can be fetched without
  processing the full 16.7 GB release.
