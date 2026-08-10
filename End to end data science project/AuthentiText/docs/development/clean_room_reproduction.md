# Clean-room reproduction audit

## Outcome

The local clean-room audit passed on 2026-08-10. A fresh clone at commit
`d52a3baa43f5a681449f9623fa8782d5d3019a6b` was created with Git for Windows'
default `core.autocrlf=true`, installed into a new CPython 3.14.6 virtual
environment, and exercised using the same dependency, evidence, Ruff, test,
and wheel commands documented for CI. The checkout remained clean after all
checks.

The source-only clone contained 151 files and 5,300,169 bytes excluding
`.git`. The environment contained 27 installed distributions and `pip check`
reported no broken requirements.

| Gate | Observed result |
| --- | --- |
| Experiment registry | 9 completed hash-linked experiments verified |
| Committed metadata | 23 JSON files verified |
| Model card | 43 evidence rows reconciled |
| Documentation | 35 Markdown files, 26 README evidence rows, and 27 generator evidence rows verified |
| Ruff lint | Passed |
| Ruff format | 118 files already formatted |
| Tests | 81 passed |
| Wheel | `authentitext-0.1.0-py3-none-any.whl`, 101,961 bytes, SHA-256 `d8980a84e26fef11fcc809ccd744c4a4c7511f77952f0f577addcd0e5dda366c` |

The machine-readable result is
[`data/metadata/clean_room_reproduction_report.json`](../../data/metadata/clean_room_reproduction_report.json).

## Procedure

The audit used a new ignored clone rather than the existing development
worktree. After cloning, the audited environment ran:

```powershell
python -m venv --without-pip .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install -r requirements\dev.lock setuptools==84.0.0
.\.venv-clean\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv-clean\Scripts\python.exe -m pip check
.\.venv-clean\Scripts\python.exe scripts\build_experiment_registry.py --check
.\.venv-clean\Scripts\python.exe scripts\check_committed_metadata.py
.\.venv-clean\Scripts\python.exe scripts\check_model_card.py
.\.venv-clean\Scripts\python.exe scripts\check_documentation.py
.\.venv-clean\Scripts\python.exe -m ruff check .
.\.venv-clean\Scripts\python.exe -m ruff format --check .
.\.venv-clean\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv-clean\Scripts\python.exe -m pip wheel --no-deps --wheel-dir build\wheels .
```

The audited Codex workspace exposes the documented CPython 3.14
temporary-directory ACL incompatibility during `ensurepip`. The audit used a
pre-created workspace-owned temporary directory to bootstrap the pinned
`pip==26.1.2`; this workaround did not alter the resolved dependency graph or
the repository. The ordinary environment setup remains appropriate outside
that sandbox.

## Portability findings

The first audit attempt found that raw SHA-256 links between committed JSON
reports changed under a CRLF checkout. Two corrections were made before the
passing audit:

- registry report identities now hash canonical UTF-8/LF content; and
- `.gitattributes` pins committed text formats to LF and explicitly marks
  compressed/model assets as binary.

The final clone confirmed `w/lf` for the overlap, external-evaluation, and
qualitative-review reports while `core.autocrlf=true` was active.

## Boundary of the result

This is a clean-room reproduction of the committed evidence surface and
package build, not a replay of the research experiments. Raw and processed
datasets, predictions, and trained artifacts are deliberately ignored by Git,
so the audit did not download MAGE or Ghostbuster, retrain models, rescore
sealed evaluations, or regenerate experiment metrics. Those deeper checks
still require the pinned inputs and artifacts named by each experiment report.

The repository has no configured remote, so no hosted Ubuntu GitHub Actions run
was observed. External documentation links were also outside this local audit.
