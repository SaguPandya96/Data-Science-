# Local FastAPI service

The API exposes the frozen inference contract without adding another scoring or
validation path. It loads the hash-verified model and calibration artifacts once
during application startup. If loading fails, liveness remains available but
readiness and prediction return HTTP 503.

## Run

```powershell
python scripts/run_api.py
```

The default binds only to `127.0.0.1:8000`. Use `--host` or `--port` explicitly
to change this. No authentication or TLS is included, so binding to a public
interface is not a supported deployment configuration.

The browser interface is served at `/`; its packaged CSS and JavaScript are
served beneath `/static`. These local assets call the same prediction endpoint
documented below and introduce no separate backend.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Local uncertainty-first browser interface |
| GET | `/health/live` | Process liveness, independent of model readiness |
| GET | `/health/ready` | Verified model/calibrator readiness and hashes |
| GET | `/v1/version` | Service, API, and inference schema versions |
| GET | `/v1/model` | Dataset revision, artifact hashes, and thresholds |
| GET | `/v1/metrics` | Process-local privacy-safe aggregate metrics |
| GET | `/v1/drift` | Validation-reference aggregate drift status |
| POST | `/v1/predict` | One inference-contract prediction |
| POST | `/v1/predict/batch` | Ordered batch of 1–32 predictions |

Single request:

```json
{
  "text": "Text to evaluate"
}
```

Batch request:

```json
{
  "texts": ["First text", "Second text"]
}
```

The batch total is capped at 200,000 characters; every item also follows the
100,000-character single-input limit. A bad item rejects the whole batch with
its zero-based index and stable item error code. Partial results are not
returned.

## Errors

Errors use:

```json
{
  "error": {
    "code": "stable_code",
    "message": "Human-readable description"
  }
}
```

Inference input failures return 422. Model-not-ready returns 503. Request-schema
errors are transformed into field location, error type, and message only; the
Pydantic `input` value is removed so malformed bodies cannot echo submitted
text. Internal traces are not included in responses.

## Privacy and logging

Successful single prediction logs contain category, character count, and
whitespace-token count. Batch logs contain item count. Request bodies, raw text,
scores, artifact feature values, and response payloads are not logged by the
application. Uvicorn's default access log records method, path, status, and
network metadata, not request bodies.

Automated tests send a unique secret through valid and invalid paths and require
it to be absent from application logs and responses. There is no database or
request persistence in this version.

`GET /v1/metrics` exposes bounded latency samples and process-level aggregate
counts and histograms. It stores no text, text hashes, identifiers, or
per-request records. See the [monitoring contract](../operations/monitoring.md)
for the complete schema semantics and privacy boundary.

`GET /v1/drift` compares those aggregates with the hash-verified development
reference after at least 760 successful items. It returns `insufficient_data`,
`within_reference`, or `investigate`; it never takes automatic action. See the
[drift contract and measured backtest](../operations/drift.md).

## Development dependencies

The runtime pins FastAPI 0.141.1 and Uvicorn 0.52.1. HTTPX2 2.9.1 is used only
by the Starlette test client and is development-only. All resolved API packages
are pinned in `requirements/runtime.lock` or `requirements/dev.lock`; `pip
check` must pass.

Startup, readiness, privacy-safe smoke checks, failure triage, and complete-set
rollback are documented in the [local operations runbook](../operations/runbook.md).
