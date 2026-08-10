# Local browser interface

AuthentiText serves a dependency-free browser interface from the same FastAPI
process as the frozen inference API. Start the service with:

```powershell
python scripts/run_api.py
```

Then open `http://127.0.0.1:8000/`. The interface calls only the local
`POST /v1/predict` endpoint. It has no analytics, third-party assets, cookies,
browser storage, or text-persistence path. Submitted text is kept in the page's
textarea for the current view and sent to the local API for the requested
prediction; clearing or reloading the page removes it from the interface.

## Interpretation

The page presents the same frozen three-way policy as the CLI and API:

- likely human-written at or below the lower calibrated threshold;
- uncertain between the two thresholds;
- likely machine-generated at or above the upper calibrated threshold.

The calibrated machine likelihood is context, not an authorship probability or
proof. Warnings, evidence quality, input counts, model hashes, and the inference
contract's limitations remain visible with the result. The frozen test metrics
shown on the page come from `data/metadata/mage_frozen_test_report.json`; they
are not estimates generated for the interface. In particular, 57.0708% of the
held-out test passages were uncertain.
The interface and every prediction repeat the canonical disclaimer in the
[responsible-AI guide](../RESPONSIBLE_AI.md); no decisive category overrides
its required human-review and prohibited-use constraints.

## Accessibility and delivery

The static HTML, CSS, and JavaScript provide a skip link, an explicit textarea
label, keyboard submission with Control/Command+Enter, visible focus states, an
announced error region, an announced and programmatically focused result, a
responsive single-column layout, and reduced-motion handling. Color is not the
only carrier of the three outcomes.

The files are package data under `src/authentitext/web`; there is no frontend
build step. This keeps the current local deployment reproducible on the audited
workstation, where Node.js is unavailable, and avoids introducing a second
service or dependency graph for a single-page local client. FastAPI applies a
self-only content security policy plus no-referrer, no-sniff, and anti-framing
headers to both the interface and API responses.

Automated tests cover the static entry point, local assets, core accessibility
hooks, privacy constraints, responsive/reduced-motion CSS, and security headers.
They do not replace assistive-technology testing or a human usability review.
