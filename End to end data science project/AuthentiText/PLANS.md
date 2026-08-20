# AuthentiText execution plan

This is a living plan. A phase is complete only after its validation criteria
are satisfied. Decisions are recorded when they are made, not in advance.

## Phase 0 — Research and problem framing

- **Objective:** Define the scientific question, candidate datasets, dataset
  roles, and evaluation regimes.
- **Tasks:** Review primary sources, compare provenance and licenses, estimate
  compute needs, select datasets, and define the untouched external test set.
- **Dependencies:** None.
- **Validation:** Sources are traceable; unknown facts stay marked unknown; the
  dataset registry and evaluation design receive a consistency review.
- **Status:** Complete.
- **Important decisions:** The detector will produce probabilistic categories,
  never authorship claims. Human false positives and generalization are primary
  evaluation concerns. English MAGE is the development corpus, with its
  WritingPrompts source excluded; Ghostbuster is the sealed external corpus;
  RAID is conditional on a storage-safe source-group sampling plan.

## Phase 1 — Development environment

- **Objective:** Establish a reproducible environment compatible with the
  selected data and modeling stack.
- **Tasks:** Choose a supported Python version, add minimal project metadata,
  pin important tools, and document setup.
- **Dependencies:** Phase 0 dependency and compute assessment.
- **Validation:** A fresh environment installs successfully and its basic checks
  run from documented commands.
- **Status:** Complete for the CPU baseline environment.
- **Important decisions:** The host has Windows build 26200, Python 3.14.6, Git
  2.55.0, 8 logical processors, 15.82 GiB RAM, Intel HD Graphics 630, and no
  detected CUDA, Docker, or Node.js. The baseline supports Python 3.12-3.14 and
  pins scikit-learn 1.9.0 with its CPU numeric stack. Local disk headroom is
  limited (7.60 GiB on C: and 13.60 GiB on D: at inspection), so full
  multi-gigabyte benchmarks must not be downloaded blindly.

## Phase 2 — Dataset acquisition

- **Objective:** Acquire selected public data reproducibly without committing it
  to Git.
- **Tasks:** Implement one source at a time, pin source versions, verify files,
  resume safely where practical, and record provenance.
- **Dependencies:** Phases 0 and 1.
- **Validation:** A clean run downloads the documented source and verifies its
  identity, schema entry point, and expected local files.
- **Status:** Complete for selected first-cycle data. The pinned MAGE OOD and
  main development partitions and the pinned Ghostbuster main external corpus
  are acquired and verified. Conditional RAID acquisition remains deferred.
- **Important decisions:** Start with MAGE's two small OOD files to validate the
  acquisition path before downloading larger development partitions. Raw data
  remains ignored; versioned manifests carry pinned URLs, checksums, expected
  sizes, and schema entry points. The main partitions total 547,760,728 bytes,
  which fits the measured local storage constraint. Ghostbuster is pinned to
  commit `86ebd72590556a81622986fab736ab9227a948af`; all 21,000 declared files
  were found and six blank upstream human essays were recorded and excluded,
  leaving 20,994 usable external records.

## Phase 3 — Data profiling and validation

- **Objective:** Quantify source quality before modeling.
- **Tasks:** Validate schema, labels, identifiers, missing and empty text,
  encoding, lengths, metadata completeness, and source balance.
- **Dependencies:** Phase 2.
- **Validation:** Automated checks and a reproducible report run on acquired
  data; failures are explicit.
- **Status:** Complete for all acquired MAGE partitions and the sealed,
  aggregate-only Ghostbuster preparation profile.
- **Important decisions:** Profiles are deterministic JSON aggregates and do
  not contain source text. MAGE raw labels are preserved (`0` machine, `1`
  human); later processing will map them to the project-wide machine-positive
  target (`1` machine, `0` human). The main partitions contain 43,609
  WritingPrompts rows to exclude and 52 exact-text groups crossing validation
  and test; these findings constrain cleaning and leakage handling.

## Phase 4 — Data cleaning

- **Objective:** Produce traceable, minimally transformed usable records.
- **Tasks:** Define normalization, filter invalid records, retain removal reasons,
  and preserve source identifiers and metadata.
- **Dependencies:** Phase 3.
- **Validation:** Cleaning is deterministic and removal counts reconcile to raw
  inputs.
- **Status:** Complete for the MAGE development partitions.
- **Important decisions:** Preserve decoded text exactly; exclude all `wp_`
  sources with a recorded reason; map MAGE raw labels to the machine-positive
  canonical target; and derive separate content and record SHA-256 identifiers.
  Store deterministic gzip JSON Lines outside Git and version only the complete
  reconciliation report.

## Phase 5 — Leakage analysis

- **Objective:** Identify record and source relationships that could inflate
  evaluation.
- **Tasks:** Check exact, normalized, and near duplicates; shared prompts and
  documents; templates; and metadata artifacts.
- **Dependencies:** Phase 4.
- **Validation:** Reusable leakage checks produce reviewed findings and proposed
  grouping constraints.
- **Status:** Complete for the first-cycle MAGE analysis and sealed external
  gate: population exact and normalized audits, the MAGE source-balanced
  high-overlap audit, and the bounded population-wide MAGE-to-Ghostbuster audit.
- **Important decisions:** Normalize with NFKC, case-folding, and whitespace
  collapsing for a second exact key while retaining the raw content ID. Treat
  conflicting normalized targets and cross-partition groups as split-policy
  inputs, not silent cleaning side effects. Published MAGE partitions are
  source-overlapping because all 288 retained source values span every split.
  The near-duplicate result is a sampled lexical audit with candidate caps, not
  a corpus-wide paraphrase measurement or population-rate estimate. The
  external gate found no exact, normalized, or confirmed 0.8 lexical
  cross-dataset pair. Exclude three redundant internal normalized Ghostbuster
  copies before scoring, leaving 20,991 external records.

## Phase 6 — Dataset splitting

- **Objective:** Create deterministic, leakage-aware evaluation partitions.
- **Tasks:** Build in-distribution, domain-holdout, generator-holdout, external,
  and robustness regimes where metadata supports them.
- **Dependencies:** Phase 5.
- **Validation:** Seeds and algorithms are recorded; groups do not cross splits;
  the external test set is sealed from development use.
- **Status:** Complete for acquired data: the leakage-sanitized MAGE
  in-distribution split and source-group holdout selectors are defined and
  validated, and Ghostbuster is materialized as a test-only external partition.
  Cross-dataset overlap remains a mandatory pre-scoring gate.
- **Important decisions:** Build connected components from all normalized-equal
  records and the 12 confirmed sampled high-overlap edges. Drop any component
  with conflicting targets. Otherwise retain one record using train, then test,
  then validation precedence and lowest record ID as the deterministic
  within-partition tie-break. Records are never moved between published
  partitions. This removes 154 rows and retains 388,919. Define nine
  leave-one-domain-out regimes and 27 leave-one-exact-generator-out regimes
  from the upstream `source` field. Store selectors and measured counts rather
  than duplicating text files.

## Phase 7 — Exploratory data analysis

- **Objective:** Test whether dataset composition or artifacts could dominate
  detector behavior.
- **Tasks:** Analyze labels, domains, generators, lengths, lexical and structural
  features, duplication, and source-specific cues.
- **Dependencies:** Phase 6.
- **Validation:** Each reported statistic or chart is generated by code and
  answers a documented question.
- **Status:** Complete for the sanitized MAGE train and validation partitions.
  Published test text and labels remain excluded from exploratory feature
  analysis.
- **Important decisions:** Generate only deterministic, text-free aggregates.
  Measure class/domain/source composition, character and whitespace-token
  lengths, and nine predefined structural indicators. The development data
  shows a material length tail difference: 12.7034% of machine records versus
  5.1012% of human records exceed 512 whitespace tokens. Add a length-only
  diagnostic baseline later to quantify this artifact, not as a product model.

## Phase 8 — Baseline modeling

- **Objective:** Establish simple, interpretable reference models.
- **Tasks:** Train majority and TF-IDF logistic-regression baselines; consider one
  additional linear baseline only if justified.
- **Dependencies:** Phases 6 and 7.
- **Validation:** Training is deterministic, configuration is recorded, and model
  artifacts reload successfully.
- **Status:** Complete for the first in-distribution baseline run.
- **Important decisions:** Train a majority baseline, a two-feature log-length
  logistic diagnostic, and a class-balanced word unigram/bigram TF-IDF logistic
  baseline. Cap the vocabulary at 100,000 features for the measured CPU and
  storage constraints. Fit only the sanitized train partition; do not inspect
  validation or test during training. Keep model files outside Git and version
  their hashes, configuration, convergence state, and measured resource facts.

## Phase 9 — Baseline evaluation

- **Objective:** Measure baseline discrimination, errors, cost, and generalization.
- **Tasks:** Report classification, ranking, false-positive, latency, and size
  metrics overall and across supported subgroups.
- **Dependencies:** Phase 8.
- **Validation:** Metrics are recomputable from saved predictions; no test data
  influenced model or threshold selection.
- **Status:** Complete for the first in-distribution baseline cycle: validation
  and a single frozen published-test run. Source-holdout and external regimes
  remain Phase 11 work.
- **Important decisions:** Persist text-free validation predictions so every
  metric can be recomputed. Report fixed-threshold classification, ranking,
  calibration, domain, generator, strategy, length-band, throughput, and model
  size evidence. The word TF-IDF model is the only baseline that materially
  exceeds chance ranking, but its 0.5 threshold has a 26.6089% human
  false-positive rate and must not be treated as a deployment threshold. On
  frozen test, calibrated abstention covers 42.9292% with 5.2391% human
  false-machine and 5.9880% machine false-human rates; preserve those misses
  without test-driven retuning.

## Phase 10 — Transformer modeling

- **Objective:** Evaluate one compute-appropriate encoder against the baselines.
- **Tasks:** Select one architecture from evidence, separate training and
  inference configuration, train, and preserve reproducibility metadata.
- **Dependencies:** Phase 9.
- **Validation:** The run completes on documented hardware, reloads for
  inference, and yields saved predictions without fabricated results.
- **Status:** Complete for the prespecified BERT-Tiny candidate. A hosted CPU
  run trained all 287,843 sanitized rows for three epochs, reload-verified the
  saved checkpoint, selected calibration and thresholds on validation only,
  froze the artifacts, and evaluated test and MAGE OOD exactly once. The
  candidate improved in-distribution results but failed the OOD gate and was
  rejected for deployment.
- **Important decisions:** Keep the transformer stack isolated from the API
  runtime. Do not run a small, non-comparable training subsample merely to fill
  the phase. The candidate still requires pretrained-weight verification,
  full-train fitting, independent calibration, and the complete
  generalization/error gate in `docs/MODEL_SELECTION.md`. Preserve the failed
  OOD result without post-test retuning.

## Phase 11 — Generalization experiments

- **Objective:** Measure performance under domain, generator, and dataset shift.
- **Tasks:** Run the defined holdouts and untouched external evaluation and
  compare them with in-distribution results.
- **Dependencies:** Phases 9 and 10.
- **Validation:** Results are generated from fixed predictions and include human
  false-positive rates and uncertainty intervals where appropriate.
- **Status:** Complete for first-cycle implemented regimes: sanitized
  in-distribution test and frozen-policy MAGE
  GPT-4/paraphrase OOD stress evaluation are complete. All nine independently
  trained and calibrated leave-one-domain-out folds are complete and verified.
  All 27 independently trained, calibrated, checkpointed, and verified
  leave-one-exact-generator-out folds are also complete. The overlap-gated
  Ghostbuster corpus was scored once with the unchanged frozen policy and its
  external result was independently verified.
- **Important decisions:** Treat the published-test result as immutable once
  scored. It is an in-distribution estimate, not evidence of source or dataset
  generalization.

## Phase 12 — Robustness experiments

- **Objective:** Measure detector sensitivity to documented edits, attacks,
  decoding variation, and truncation.
- **Tasks:** Evaluate published benchmark perturbations and length conditions
  without publishing evasion instructions.
- **Dependencies:** Phases 9 and 10; a suitable robustness dataset.
- **Validation:** Conditions and sample counts are traceable to source metadata;
  results are reported even when poor.
- **Status:** Complete for the locally available first-cycle evidence: MAGE's
  paraphrase stress set, natural length bands, and paired 50-, 100-, and
  200-token prefix truncations are evaluated. Other documented edit, attack,
  and decoding conditions are deferred pending suitable published data.
- **Important decisions:** Keep the policy frozen and report original GPT-4,
  GPT-4 paraphrase, and machine-paraphrased-human outcomes separately. Do not
  claim paired effects because upstream supplies no stable pair identifier.
  Deduplicate the 762 repeated human controls for combined metrics. For the
  separate truncation intervention, preserve each stable test record ID and
  compare the complete original with its deterministic prefix on the same
  eligible row. Fix all budgets before viewing outcomes and prohibit result-
  driven model, calibration, or threshold changes.

## Phase 13 — Calibration and uncertainty

- **Objective:** Turn scores into empirically evaluated probabilities and
  abstaining decisions.
- **Tasks:** Compare justified calibration methods, measure Brier score and ECE,
  and derive thresholds from validation data under a human false-positive goal.
- **Dependencies:** Phases 9 through 12.
- **Validation:** Calibration and thresholds use validation data only and are
  evaluated on untouched tests.
- **Status:** Complete for the word TF-IDF baseline; any future candidate model
  requires its own calibration cycle.
- **Important decisions:** Hash-assign validation records to 40% calibration
  fit, 30% method/threshold selection, and 30% untouched calibration audit.
  Compare raw, sigmoid, and isotonic probabilities by selection Brier score,
  then ECE, then method name. Isotonic wins. Use thresholds 0.231884057971 and
  0.717391304348 to create likely-human, uncertain, and likely-machine outputs,
  targeting 5% cross-class decisive error rates on policy selection. Preserve
  the roughly 60% abstention rate rather than forcing high-confidence labels.

## Phase 14 — Error analysis

- **Objective:** Understand systematic high-cost failures.
- **Tasks:** Review high-confidence false positives and negatives, uncertain
  cases, length effects, and domain, generator, external, and edit failures.
- **Dependencies:** Phase 13.
- **Validation:** Findings link to reproducible record identifiers and small,
  license-conscious excerpts where needed.
- **Status:** Complete for the first-cycle baseline; quantitative validation,
  in-distribution test, MAGE OOD, nine domain-holdout, and 27
  exact-generator-holdout analyses cover
  length, domain, exact generator, calibration, abstention, and cross-class
  decisive errors. Ghostbuster adds external domain, generator, and prompt
  strategy evidence. A deterministic 21-record external review covers
  score-extreme human false-machine and machine false-human cases plus
  target/domain-stratified uncertain-boundary cases.
- **Important decisions:** Preserve text-free prediction artifacts for
  reproducibility. Prioritize the measured human false-machine failures and
  paraphrase sensitivity; do not cherry-pick favorable aggregate metrics. Keep
  excerpts local and commit only stable IDs, coded surface cues, generic notes,
  counts, and limitations. Treat the single-reviewer purposive sample as
  qualitative evidence, not a population estimate or causal explanation.

## Phase 15 — Model selection

- **Objective:** Choose a candidate from evidence rather than one aggregate
  score.
- **Tasks:** Compare generalization, human false positives, calibration, latency,
  memory, size, and maintainability; document the decision.
- **Dependencies:** Phases 11 through 14.
- **Validation:** `docs/MODEL_SELECTION.md` cites actual experiment artifacts and
  records tradeoffs.
- **Status:** Complete for first-cycle implemented candidates.
- **Important decisions:** Retain word TF-IDF logistic plus isotonic calibration
  and abstention as the local research baseline because it is the only
  implemented model materially above chance and fits the audited workstation.
  Reject the majority and length controls. Do not approve the selected baseline
  for production or high-stakes use because it misses both frozen decisive-error
  point targets and degrades severely by domain, length, and MAGE OOD condition.
  Transformer performance remains unknown, not assumed.

## Phase 16 — Inference pipeline

- **Objective:** Implement a versioned text-to-decision contract.
- **Tasks:** Load model and calibration artifacts, handle short and long text
  explicitly, apply thresholds, and return limitations.
- **Dependencies:** Phase 15.
- **Validation:** Unit and integration tests cover loading, preprocessing,
  segmentation, aggregation, and decision boundaries.
- **Status:** Complete for the frozen baseline.
- **Important decisions:** Load the base model and calibration policy only after
  size, hash, type, and linkage checks. Accept one nonblank Unicode string up to
  100,000 characters and reject NUL. Return calibrated likelihood, frozen
  category, threshold/model provenance, input counts, warnings, and limitations
  without returning or persisting input text. Warn—without post-test category
  overrides—below 50 whitespace tokens and for formatting absent from the
  development profile.

## Phase 17 — FastAPI service

- **Objective:** Expose validated inference through a stable local API.
- **Tasks:** Add health, readiness, version, model, single prediction, and batch
  endpoints with request-size and error controls.
- **Dependencies:** Phase 16.
- **Validation:** API tests cover valid, empty, Unicode, oversized, batch, and
  failure cases without leaking stack traces.
- **Status:** Complete for local service version 1.
- **Important decisions:** Load and verify the predictor during application
  lifespan; keep liveness independent from model readiness. Expose version,
  model, single-prediction, and bounded-batch endpoints. Limit batches to 32
  items and 200,000 total characters. Sanitize Pydantic errors so submitted
  values are not echoed, log only category/count metadata, and return no stack
  traces or text.

## Phase 18 — Frontend

- **Objective:** Provide an accessible interface that communicates uncertainty
  and limitations clearly.
- **Tasks:** Implement text entry, loading, error, result, likelihood, and
  limitations states with responsive and keyboard-accessible behavior.
- **Dependencies:** Stable Phase 17 response contract.
- **Validation:** Automated tests cover the HTML entry point, local assets,
  privacy constraints, security headers, responsive layout hooks, reduced
  motion, and core accessibility semantics. The package build contains all
  static assets. Manual assistive-technology review remains future work.
- **Status:** Complete for the local version 1 interface.
- **Important decisions:** Serve dependency-free HTML, CSS, and JavaScript from
  the existing FastAPI process because the audited workstation has no Node.js
  runtime and the product does not need a second service. Make abstention,
  warnings, limitations, and frozen test evidence prominent. Use no analytics,
  third-party assets, cookies, browser storage, or text persistence. Apply a
  self-only content security policy and related browser hardening headers.

## Phase 19 — Operational metadata

- **Objective:** Record useful service signals without retaining submitted text
  by default.
- **Tasks:** Define request metadata, hashes, versions, latency, results, and
  errors; decide whether persistence is justified.
- **Dependencies:** Phase 17.
- **Validation:** Tests prove raw text is not persisted or logged under default
  configuration.
- **Status:** Complete for process-local version 1 metadata.
- **Important decisions:** Application logs contain category and input counts
  for successful requests, never submitted text. Keep monitoring in memory with
  no database or per-request records. Do not compute text hashes because they
  remain linkable and vulnerable to dictionary matching. Expose only fixed
  endpoint/status/error counters, a bounded latency window, aggregate
  length/score/outcome/warning distributions, and verified versions/hashes.

## Phase 20 — Docker

- **Objective:** Package existing services for a simple reproducible local run.
- **Tasks:** Add minimal images and Compose configuration after service boundaries
  exist.
- **Dependencies:** Phases 17 and 18; Docker availability.
- **Validation:** Images build and the documented Compose workflow passes health
  and prediction smoke tests.
- **Status:** Complete. The minimal local image packages the API with a non-root
  user, health check, read-only artifact mount, and restricted runtime. Its
  hosted GitHub build passed. A separate deployment image retrieves the frozen
  release artifacts and verifies their byte counts and SHA-256 identities at
  build time; its hosted build remains the final deployment-package gate.
- **Important decisions:** Keep trained artifacts outside the image and mount
  them read-only so startup identity checks remain authoritative. Kubernetes is
  out of scope without a concrete need.

## Phase 21 — Testing

- **Objective:** Maintain risk-proportionate coverage across the evolving system.
- **Tasks:** Extend focused data, feature, model, API, and integration tests with
  each implementation phase.
- **Dependencies:** Continuous across all implementation phases.
- **Validation:** The full documented test suite passes in a clean environment;
  tests assert behavior rather than coverage counts alone.
- **Status:** Complete for local version 1. All 81 tests passed in a fresh clone
  at `d52a3baa43f5a681449f9623fa8782d5d3019a6b`; the first hosted run remains
  unobserved because the repository has no configured remote.
- **Important decisions:** Testing is incremental, not a final cleanup phase.
  Use the standard-library `unittest` runner until a demonstrated need
  justifies another test dependency. Ruff enforces lint and formatting checks.
  Committed report checks cover JSON integrity, stored validation markers, and
  frozen artifact linkage without pretending to reproduce ignored experiments.

## Phase 22 — CI/CD

- **Objective:** Automate lightweight repository checks without retraining large
  models on every change.
- **Tasks:** Add install, format, lint, type, unit, integration, and image-build
  checks as those capabilities become real.
- **Dependencies:** Stable Phase 1 tooling and enough executable code.
- **Validation:** The workflow passes from a clean checkout and has documented
  artifact and caching behavior.
- **Status:** Configured and clean-room validated locally. Every documented
  command, including the ordinary wheel build, passed from a fresh clone. The
  first hosted execution is not observed because the repository has no
  configured remote and no push was performed.
- **Important decisions:** Use CPython 3.14 with both dependency locks in the
  cache key, read-only repository permissions, disabled persisted checkout
  credentials, and a bounded runtime. Run install, dependency, metadata, lint,
  format, unit/integration, and wheel-build checks. Do not download datasets,
  retrain, or rerun frozen evaluations in routine CI. Type, container, and
  deployment jobs remain absent until those capabilities are implemented and
  locally verifiable.

## Phase 23 — Monitoring

- **Objective:** Observe service health and prediction-distribution signals.
- **Tasks:** Measure counts, errors, latency quantiles, lengths, classifications,
  confidence, uncertain rate, and model version without fake traffic.
- **Dependencies:** Phases 17 and 19.
- **Validation:** Metrics tests and a local smoke run verify values from real or
  explicitly labeled fixture requests.
- **Status:** Complete for local aggregate monitoring; drift decisions and
  production alerting remain separate later phases.
- **Important decisions:** The snapshot begins empty and reflects actual
  process traffic only. Use a 2,048-request bounded latency window with
  nearest-rank quantiles and process-lifetime aggregate counters. Reading
  metrics does not increment them. Fixed allowlists prevent input-derived or
  unbounded-cardinality labels. No metric triggers retraining.

## Phase 24 — Drift detection

- **Objective:** Flag meaningful input or prediction distribution changes for
  investigation.
- **Tasks:** Define reference windows and tests for length, embedding, prediction,
  confidence, language, and potential domain drift.
- **Dependencies:** Phase 23 and sufficient real observations.
- **Validation:** Backtests on documented shifts measure sensitivity and false
  alerts; alerts never trigger automatic retraining.
- **Status:** Complete for the four process-local aggregate signals.
- **Important decisions:** Build the reference only from the 50,509 sanitized
  validation rows, stored validation predictions, and frozen calibrator; do not
  read test data. Select total-variation thresholds on 40 deterministic hash
  windows and audit on 20 disjoint windows. Require at least 760 observations.
  The actual audit flagged 1/20 same-distribution windows (5%), while all 9/9
  real validation-domain shifts were flagged. Treat every flag as an
  investigation prompt only. Never update thresholds, alert externally, or
  retrain automatically.

## Phase 25 — Deployment

- **Objective:** Deploy the verified system with explicit cost and operational
  constraints.
- **Tasks:** Select a target only after resource measurements, define secrets and
  rollback handling, and run acceptance checks.
- **Dependencies:** Phases 20 through 24.
- **Validation:** A documented deployment passes health, readiness, prediction,
  privacy, monitoring, and rollback checks.
- **Status:** Complete for the free portfolio deployment. The Render Blueprint
  deployed commit `9556360` at `https://authentitext-tsaq.onrender.com`. All
  required health, version, model, prediction, monitoring, and drift endpoints
  returned HTTP 200; the live artifact hashes matched the frozen release. The
  text-free acceptance evidence is committed in
  `data/metadata/render_deployment_acceptance_report.json`.
- **Important decisions:** Default to free or local infrastructure; no paid
  resource may be created without explicit permission.

## Phase 26 — Final technical documentation

- **Objective:** Make the complete system reproducible and its limitations clear.
- **Tasks:** Finalize the README, data lineage, decision log, experiment log,
  model card, responsible-AI guidance, retraining design, and operations docs.
- **Dependencies:** All relevant implementation and evaluation phases.
- **Validation:** A clean-room reproduction review follows the documentation,
  reconciles every machine-checked evidence row to committed reports, verifies
  tests and packaging, and records which ignored artifacts prevent experiment
  replay.
- **Status:** Complete for the version 1 committed evidence surface. The model
  card, responsible-AI guidance, retraining design, experiment log, operations
  runbook, and README are evidence-checked. A fresh-clone audit passed all
  dependency, evidence, Ruff, test, and packaging gates; it explicitly did not
  replay experiments whose pinned data and artifacts are ignored by Git.
- **Important decisions:** Documentation evolves with verified behavior; it will
  not claim unfinished work. Model-card evidence rows are reconstructed from
  committed reports in CI so reported values and frozen artifact identities
  cannot drift silently. Preserve the machine-readable clean-room report and
  its scope boundary instead of treating a package-level audit as new model
  evidence.
