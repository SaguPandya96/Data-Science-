# SupplyLens Build Log

This log records major build and verification commands, their observed outcomes, and any corrective action. Timestamps use America/Los_Angeles calendar dates.

## 2026-08-01

| Stage | Command | Result | Notes |
|---|---|---|---|
| Workspace inspection | `Get-ChildItem -Force` | Succeeded | The generated parent folder contained only `outputs/` and `work/`; the repository was created in a new `SupplyLens/` directory. |
| Toolchain inspection | `git --version`; `python --version`; `py -0p` | Succeeded | Git 2.53.0; Python 3.11 is installed and selected for the project. |
| Repository initialization | `git init -b main SupplyLens` | Succeeded | Created an empty Git repository on branch `main`. |
| Environment creation | `py -3.11 -m venv SupplyLens/.venv` | Succeeded | Created an isolated Python 3.11 environment. |
| Structure initialization | PowerShell `New-Item` commands for required directories | Succeeded | Created configuration, data, notebook, package, script, app, test, model, report, documentation, and CI directories. |

| Dependency installation | `.venv\\Scripts\\python -m pip install -r requirements.txt` | Failed in three attempts | The package resolver stalled before completing; no successful installation was claimed. |
| Dependency workaround | Workspace-local `uv pip install -r requirements.txt` and `uv pip install -e .` | Succeeded | Installed the exact pinned Python 3.11 dependencies into `.venv`; direct version imports matched the pins. |
| Data acquisition | `.venv\\Scripts\\python scripts\\download_data.py` | Succeeded | Downloaded 3,785,904 bytes from the commit-pinned public mirror and verified SHA-256 `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673`. |
| Raw and processed validation | `.venv\\Scripts\\python scripts\\validate_data.py` | Succeeded with retained warnings | Validated 10,324 source lines and 33 columns, then produced 7,030 shipment records. Retained three recorded-before-delivery rows and four more-than-365-days-early rows as warnings. |
| Initial training evaluation | `.venv\\Scripts\\python scripts\\train.py` | Failed | Isotonic probability ties caused the threshold implementation to review more rows than the exact capacity table. This exposed an evaluation-policy defect, not a model-training defect. |
| Capacity-policy correction | Replaced threshold-only tie handling with stable exact-rank selection | Succeeded | The same stable ranking rule is now used for metrics and scoring-capacity evaluation. |
| Final training | `.venv\\Scripts\\python scripts\\train.py` | Succeeded | Trained baselines, logistic regression, histogram gradient boosting, calibration candidates, lead-time experiments, causal-readiness diagnostics, and operational outputs. Serialized the selected bundle locally. |
| Report generation | `.venv\\Scripts\\python scripts\\build_reports.py` | Succeeded | Generated 15 figures, 26 tables, and machine-readable metric artifacts. |
| README generation | `.venv\\Scripts\\python scripts\\build_readme.py` | Succeeded | Rendered headline results from `reports/metrics/final_metrics.json`. |
| Dashboard launch | `.venv\\Scripts\\python -m streamlit run app\\app.py` | Initially failed in the browser | The installed dashboard version rejected path objects passed to image rendering. |
| Dashboard correction and visual QA | Converted figure paths to strings and reloaded all seven dashboard sections | Succeeded | Executive, queue, supplier, model, segment, monitoring, and limitation views loaded. A legitimate 2,880 by 1,800 screenshot was captured. |
| Lint | `.venv\\Scripts\\python -m ruff check .` | Initially failed | Reported import ordering, unused imports, and Windows string-escape warnings. |
| Lint correction | `.venv\\Scripts\\python -m ruff check . --fix`; then `.venv\\Scripts\\python -m ruff check .` | Succeeded | Ruff fixed 41 issues; the verification run reported no remaining issues. |
| Automated tests | `.venv\\Scripts\\python -m pytest -q` | Succeeded | 19 tests passed in 71.23 seconds against the downloaded public data. |
| First notebook execution | `.venv\\Scripts\\python scripts\\build_notebook.py` | Failed | The restricted Windows environment denied the Jupyter kernel connection-file ACL operation. No cell execution was reported as successful. |
| Notebook execution fallback | Set `SUPPLYLENS_RESTRICTED_WINDOWS_NOTEBOOK=1` for the restricted local run, then reran the builder | Succeeded | All 28 code cells executed from a clean kernel with zero error outputs; the HTML export was created. Normal project execution retains secure connection-file writes. |
| Scoring CLI | `.venv\\Scripts\\python scripts\\score.py --input data\\processed\\scoring_input_verification.csv --output data\\processed\\scoring_output_verification.csv` | Succeeded | Scored 37 real shipment rows, produced exact ranks, and flagged 8 rows at 20% ceiling capacity; probabilities were finite and within `[0, 1]`. |
| Repository audit | `.venv\\Scripts\\python scripts\\validate_project.py` | Succeeded | Verified required structure, source checksum, model and report artifacts, executed notebook, metrics-generated README, file sizes, repository-relative paths, secret patterns, and excluded phrases. |
| Background dashboard health attempt | PowerShell `Start-Process` launch plus health probe | Blocked by environment policy | The environment rejected background-process launch; no dashboard result was inferred from this attempt. |
| Direct dashboard health check | Direct Streamlit server launch, then `/_stcore/health` probe | Succeeded | Returned HTTP 200 with body `ok`; the server was terminated after the check. This supplements the earlier full browser inspection. |
| Final automated test rerun | `.venv\\Scripts\\python -m pytest -q` | Succeeded | 19 tests passed in 64.04 seconds after the notebook, scoring, and dashboard verification work. |
| Final lint and project audit | `.venv\\Scripts\\python -m ruff check .`; `.venv\\Scripts\\python scripts\\validate_project.py`; `git diff --check` | Succeeded | Lint, repository validation, and whitespace checks all passed. |
| Publication capability check | `git remote -v`; `gh auth status` availability check | External publication unavailable | No remote was configured and the hosting CLI was not installed. This does not affect the local repository; publication commands are documented in the final handoff. |
| Release package generation | `git archive` and `git bundle create` to the deliverables directory | Succeeded | Created a source archive and a restorable repository bundle outside the repository, then copied the legitimate dashboard preview beside them. |
| Pre-release Git check | `git status --porcelain`; `git log --oneline` | Succeeded | The working tree was clean before this verification record was added; seven logical commits were present at that point. |

All reported success states above correspond to executed commands; original failed attempts remain in the log.
