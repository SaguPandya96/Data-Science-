# AuthentiText engineering guidelines

## Project contract

- Treat AuthentiText as a research system that estimates patterns associated
  with machine-generated text. Never present a result as proof of authorship.
- Preserve the three-way output contract: `likely_human`, `uncertain`, and
  `likely_machine`. Do not remove abstention or weaken the published warnings.
- Keep the word TF-IDF logistic model and its isotonic calibration policy as the
  version 1 runtime. The evaluated BERT-Tiny model is research evidence, not a
  production dependency.

## Code boundaries

- Put reusable application code under `src/authentitext` and command-line entry
  points under `scripts`.
- Keep the API runtime lightweight. Transformer dependencies belong only in the
  isolated Python 3.11 environment defined by `requirements/transformer.lock`.
- Maintain compatibility with the Python range declared in `pyproject.toml`.
- Avoid new frameworks or services unless they solve a measured requirement.

## Data and experiment integrity

- Treat committed files in `data/metadata` as the authoritative, text-free
  evidence record. Raw data, trained weights, and prediction files remain
  outside Git unless a release explicitly publishes them.
- Preserve dataset identifiers, revisions, licenses, row counts, byte sizes,
  SHA-256 hashes, label mappings, and transformation reports.
- Keep train, validation, test, OOD, and external roles separate. Never train,
  calibrate, select thresholds, or choose a model using test or OOD outcomes.
- Check exact, normalized, grouped, and near-duplicate leakage before changing
  a split.
- Freeze model, calibration, and threshold identities before final evaluation.
  Do not retune a completed evaluation cycle.
- Record only measurements produced by an executed and verified workflow.

## Privacy and security

- Never log, persist, or include submitted text in monitoring or evaluation
  reports. Do not add text hashes or per-request records.
- Keep credentials, tokens, local configuration, and private endpoints out of
  source control and command output.
- Validate user input at the API boundary and keep public error responses free
  of internal paths and implementation details.
- Verify model and calibration hashes and their linkage before reporting the
  service as ready.

## Implementation quality

- Prefer small, typed, deterministic functions with explicit failure modes.
- Write generated reports through a temporary file and atomically replace the
  destination only after validation succeeds.
- Add focused tests for behavior, edge cases, privacy properties, artifact
  verification, and failure paths.
- Update the README, model card, decision record, operations documentation, and
  machine-readable metadata whenever behavior or verified evidence changes.
- Keep commits cohesive and use concise technical commit messages.

## Required checks

Run the relevant subset while developing and the complete suite before merging:

```text
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest tests -q
python scripts/check_committed_metadata.py
python scripts/check_documentation.py
python scripts/check_model_card.py
python scripts/build_experiment_registry.py --check
```

Inspect the final diff and confirm that no raw text, credentials, large model
artifacts, temporary files, or unrelated changes are included.

## Deployment

- Keep runtime artifacts outside the standard container image. The Render image
  may download only the pinned public release assets and must verify their size
  and SHA-256 identity during the build.
- Preserve the non-root container user, readiness health check, disabled access
  logs, and read-only runtime assumptions.
- A deployment is acceptable only when liveness, readiness, version, model,
  prediction, metrics, and drift endpoints pass and the observed artifact hashes
  match the frozen release.
- Treat drift alerts as investigation signals only. Never trigger automatic
  retraining, threshold changes, or model promotion.
