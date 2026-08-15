# Repository guidance

- Build incrementally. Create files and infrastructure only when the current
  task requires them.
- Never fabricate data, labels, citations, metrics, findings, or experiment
  outputs. Report only commands and results that were actually run.
- Prefer credible public data, preserve its provenance and license information,
  and make every transformation reproducible.
- Protect the final external test set. Never train, tune, calibrate, or select
  thresholds or models on it.
- Check exact, normalized, and near-duplicate leakage before finalizing splits.
- Start with the simplest adequate solution and justify additional complexity
  with evidence.
- Add tests that verify meaningful behavior as functionality is introduced.
- Run relevant tests and checks, then inspect status and diffs before committing.
- Commit one coherent change at a time. Stage files deliberately and use short,
  natural, descriptive commit messages.
- Do not push, publish, open pull requests, or spend money without explicit
  permission.
- Update documentation when behavior, data lineage, decisions, or limitations
  materially change.
- Do not claim a feature works unless it was verified.
- Do not add unnecessary frameworks, premature placeholder architecture,
  interview material, resume material, or recruiter content.
- Keep raw datasets, model weights, credentials, local configuration, and large
  generated artifacts out of Git.
