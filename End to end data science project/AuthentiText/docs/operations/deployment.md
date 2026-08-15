# Render deployment

The public deployment uses the frozen lexical version 1 model. It does not use
the transformer candidate until that candidate has passed the full evaluation
and model-selection gate.

The Render image differs from the local Compose image in one deliberate way:
it downloads the two runtime artifacts from the
[`authentitext-baseline-v1`](https://github.com/SaguPandya96/Data-Science-/releases/tag/authentitext-baseline-v1)
GitHub release while the image is built. The download is restricted to HTTPS on
`github.com`; both files must match the recorded byte counts and SHA-256 hashes
before they are moved into place. Application startup repeats the artifact type,
identity, calibration-linkage, and threshold checks.

The repository-level `render.yaml` selects the free Docker web-service plan,
the monorepo project directory, and `/health/ready` as the deployment gate. No
database, persistent disk, secret, or paid instance is requested. The service
binds to Render's `PORT` value and disables Uvicorn access logs so submitted text
cannot appear in request logs.

## Create the service

1. Merge the verified deployment change into `master`.
2. In Render, choose **New > Blueprint** and connect
   `SaguPandya96/Data-Science-`.
3. Accept the repository's `render.yaml` with the instance type still set to
   **Free**.
4. Wait for `/health/ready` to pass before using the prediction endpoint.

The free service is a portfolio demonstration, not a high-availability
production system. It can sleep when idle, has an ephemeral filesystem, and
does not provide autoscaling. The model files are immutable image contents, so
the service does not need persistent storage.

## Acceptance checks

Record the deployed URL and run:

```powershell
$baseUrl = "https://YOUR-SERVICE.onrender.com"
Invoke-RestMethod "$baseUrl/health/live"
Invoke-RestMethod "$baseUrl/health/ready"
Invoke-RestMethod "$baseUrl/v1/version"
Invoke-RestMethod "$baseUrl/v1/model"
Invoke-RestMethod "$baseUrl/v1/predict" -Method Post -ContentType "application/json" -Body '{"text":"A short deployment smoke test."}'
Invoke-RestMethod "$baseUrl/v1/metrics"
Invoke-RestMethod "$baseUrl/v1/drift"
```

The acceptance record must include the deployment commit, image build result,
base-model and calibration hashes returned by readiness, response status for
each endpoint, and the rollback target. Do not include submitted text or full
prediction responses in the record.

## Rollback

Use Render's rollback control to select the last known-good deploy. Readiness
must return the expected artifact hashes before traffic resumes. If no prior
deploy exists, suspend the service rather than serving from an unverified
artifact set.
