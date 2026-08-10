# Local operations runbook

## Support boundary

This runbook covers one local AuthentiText process using the frozen version 1
artifacts. The supported default bind is `127.0.0.1:8000`. The service has no
authentication, TLS, multi-process aggregation, persistent monitoring,
background alert delivery, model registry, or remote storage. Do not expose it
to a public network.

Docker is unavailable on the audited workstation, no image has been built, and
no deployment target has been selected. This is therefore a local runbook, not
a production operations or deployment claim.

## Preflight

From the repository root, create and validate the locked environment as
documented in the [environment guide](../development/environment.md). Then run:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\build_experiment_registry.py --check
.\.venv\Scripts\python.exe scripts\check_committed_metadata.py
.\.venv\Scripts\python.exe scripts\check_model_card.py
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

The service also requires these ignored local artifacts:

```text
artifacts/baselines/id/word_tfidf_logistic.joblib
artifacts/baselines/id/calibration_policy.joblib
```

Their byte sizes and SHA-256 identities must match
`data/metadata/mage_baseline_training_report.json` and
`data/metadata/mage_calibration_report.json`. Startup performs those checks,
verifies model types and calibrator linkage, and refuses readiness on a
mismatch. Do not rename an unrelated artifact into these paths or bypass the
checks.

Optional deeper local verification requires the ignored validation predictions
and source split in addition to the model artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\train_baselines.py --verify-only
.\.venv\Scripts\python.exe scripts\calibrate_baseline.py --verify-only
```

## Start and stop

Start the service in the foreground:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py
```

Open `http://127.0.0.1:8000/` for the local interface or
`http://127.0.0.1:8000/docs` for the interactive API schema. Keep the process in
the foreground so startup and request errors remain visible. Stop it with
Control+C and confirm the process exits. A restart clears every process-local
metric and drift observation; it does not preserve an observation window.

`--host` and `--port` exist for explicit local configuration. Binding to
`0.0.0.0` or another externally reachable interface is unsupported because the
service has no authentication or TLS.

## Health and readiness

Liveness proves only that the process can answer. Readiness proves that the
model and calibrator loaded and passed identity checks.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/version
Invoke-RestMethod http://127.0.0.1:8000/v1/model
```

Do not send predictions unless readiness returns HTTP 200. Record the base
model hash, calibration hash, dataset revision, and thresholds from readiness
or `/v1/model` when investigating an incident. Liveness can remain HTTP 200
while readiness is HTTP 503; that is intentional so artifact failures are
distinguishable from process failure.

## Privacy-safe prediction smoke check

Use a clearly labeled operational fixture, never a benchmark or private user
text:

```powershell
$smokeBody = @{
  text = "Explicit local operations smoke-test fixture; not research or evaluation data."
} | ConvertTo-Json

$smokeResult = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/predict `
  -ContentType "application/json" `
  -Body $smokeBody

$smokeResult.category
$smokeResult.model
$smokeResult.limitations
```

Success means the response follows the contract, includes the canonical
disclaimer and verified model identity, and does not echo the submitted text.
Any of the three categories is a valid smoke result. The fixture is not labeled
research data, must not be added to evaluation metrics, and is not evidence of
detector quality.

## Metrics and drift

Read the process-local aggregate snapshot and drift comparison with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/metrics
Invoke-RestMethod http://127.0.0.1:8000/v1/drift
```

Reading either endpoint does not increment request counters. Metrics retain no
text, text hashes, request identifiers, or per-request records. They cover only
the current process lifetime, and latency quantiles use a bounded recent
window. Multiple processes would have separate, non-combinable snapshots.

Drift remains `insufficient_data` below 760 successful prediction items. At or
above that boundary it can return `within_reference` or `investigate` for four
aggregate distributions. `investigate` is not proof of model failure or
authorship and must not change a threshold, restart the observation window,
retrain, or promote a model automatically. Follow the
[drift contract](drift.md) and [retraining policy](retraining.md).

## Logs and privacy boundary

The application logs successful category and coarse input counts, batch item
counts, stable error codes, and startup failures. Uvicorn access logs add
method, path, status, and network metadata. Request bodies, submitted text,
scores, features, excerpts, and response bodies are not logged by the
application.

The boundary does not cover shell history, source files, the browser, host or
proxy logs, backups, screen capture, or tools placed in front of the process.
Do not add a proxy, telemetry exporter, raw debugging log, database, or feedback
collector without a new privacy/data-flow review and explicit retention and
deletion rules.

## Failure triage

### Process unavailable

Confirm the foreground process is running and the intended local port is not in
use. Inspect startup output. Do not solve a bind failure by exposing the service
on an unreviewed public interface.

### Live but not ready

Read `/health/ready` and the startup log. Check that both artifact paths exist,
then run the artifact `--verify-only` commands. A missing, wrong-size,
wrong-hash, wrong-type, or mismatched calibrator must remain a hard failure. Do
not serve an uncalibrated fallback.

### Prediction returns 422

Use the stable error code to check blank input, type, NUL, item index, single
text size, batch count, or batch-total size. Error responses intentionally omit
submitted values. Do not enable raw-body logging to diagnose a user input.

### Prediction returns 503

Treat it as the same verified-model readiness failure. Stop sending traffic and
repair the artifact installation; retries do not correct a hash or linkage
error.

### Drift reference invalid

Confirm that `data/metadata/mage_drift_reference.json` is present and that its
model/calibrator hashes match `/v1/model`. Rebuild or verify it only from the
pinned validation inputs. Never substitute the published test or current
traffic as a convenient reference.

### Unexpected errors or privacy concern

Stop the affected use, preserve only privacy-safe logs plus exact code/report
and artifact identities, and reproduce with a non-sensitive fixture. Do not
publish submitted text in an issue. Follow the incident sequence in the
[responsible-AI policy](../RESPONSIBLE_AI.md).

## Artifact change and rollback

There is no hot reload or approved version 1 promotion command. Stop the
process before any artifact change. An approved future release must install the
complete hash-linked model, calibrator, thresholds, reports, dependency lock,
and package as one version; mixing components is forbidden.

Keep the prior complete verified set available until acceptance checks pass.
Rollback restores that whole set and restarts the process, then repeats
liveness, readiness, identity, prediction, privacy, metrics, and drift checks.
Rollback does not delete the failed candidate's reports. The complete evidence
gate is defined in the [retraining design](retraining.md).

## Executed local smoke evidence

On 2026-08-09, a process-local TestClient smoke run loaded the default frozen
artifacts and sent one explicitly labeled operational fixture. It observed:

| Check | Observed result |
| --- | --- |
| Interface, live, ready, version, model, metrics, and drift GETs | HTTP 200 for all seven |
| Single fixture prediction | HTTP 200; submitted text absent from the response |
| Successful-item aggregate after prediction | 1 |
| Drift status after prediction | `insufficient_data` |

This is a real local smoke result, not fabricated production traffic, a load
test, a Docker test, a deployment test, or evidence about model accuracy.
