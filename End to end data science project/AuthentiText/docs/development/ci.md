# Continuous integration

The GitHub Actions workflow runs on every push and pull request using Ubuntu
and CPython 3.14. It installs the exact development lock, checks dependency
integrity, validates committed metadata, runs Ruff lint and format checks,
executes the complete `unittest` suite, verifies that the project builds as a
wheel, and builds the local-service container image.

The workflow grants only `contents: read`, disables persisted checkout
credentials, and has a 20-minute timeout. It pins the immutable release commits
for
[`actions/checkout` 7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1)
and
[`actions/setup-python` 7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0),
which were checked against their official release pages on 2026-08-09.

## Scope and limitations

CI starts from a fresh checkout. Raw datasets, processed text, saved
predictions, and trained model artifacts are intentionally ignored by Git, so
the workflow does not download data, train models, rebuild drift thresholds, or
repeat frozen test and OOD evaluations. The unit and integration tests use
small explicit fixtures.

`scripts/check_committed_metadata.py` checks every committed JSON metadata file
for valid finite JSON, schema version, successful stored validation status, and
consistent frozen model/calibrator hashes across calibration, test, OOD, and
drift records. This is an internal-consistency check, not experiment
reproduction and not fresh evidence that the recorded model metrics still
hold. Full experiment verification requires the pinned ignored inputs and
artifacts documented by the corresponding report.

`scripts/check_model_card.py` separately reconstructs every evidence-table row
in the model card from those reports and checks the required limitations. It
does not treat the prose or stored validation markers as a new experiment.

`scripts/build_experiment_registry.py --check` reconstructs the experiment
registry from twelve completed report files, including their SHA-256 identities,
and fails if the committed registry differs. Entries marked unrun remain
explicitly metric-free.

`scripts/check_documentation.py` resolves every local Markdown link and
reconstructs the README's data, artifact, threshold, result, drift, and
experiment-count evidence from committed reports. External links remain a
separate network review.

The pip download cache is keyed by both lock files. No datasets, trained models,
predictions, or build outputs are cached or uploaded. The wheel is built only
to validate packaging and is discarded with the runner.

Type checking, deployment, and model retraining remain absent because those
verified capabilities or tools do not yet exist in the project. The container
step proves only that the image builds; it does not supply ignored model
artifacts or perform a deployment acceptance test.

## Local reproduction

After following the environment setup guide, run the same checks from the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\build_experiment_registry.py --check
.\.venv\Scripts\python.exe scripts\check_committed_metadata.py
.\.venv\Scripts\python.exe scripts\check_model_card.py
.\.venv\Scripts\python.exe scripts\check_documentation.py
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m pip wheel --no-deps --wheel-dir build\wheels .
```

Every command above passed in a fresh Windows clone at commit
`d52a3baa43f5a681449f9623fa8782d5d3019a6b`: `pip check`, all four evidence
checks, Ruff, 81 tests, and the ordinary `pip wheel` command. The exact
environment and wheel identity are recorded in the
[clean-room reproduction audit](clean_room_reproduction.md).

The audited workspace required the documented CPython 3.14 `ensurepip`
temporary-directory shim during virtual-environment bootstrap. That local
environment condition is separate from the CI commands and dependency graph.
The clean-room audit used a standalone checkout with no configured remote, so
it did not include a hosted Ubuntu workflow run.
