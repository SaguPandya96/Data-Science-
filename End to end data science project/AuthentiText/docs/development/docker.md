# Docker workflow

The container packages the existing FastAPI service and committed text-free
metadata. Model artifacts remain ignored by Git and outside the image. Compose
mounts `artifacts/baselines/id` read-only at runtime; startup still verifies the
recorded byte sizes, SHA-256 identities, artifact types, calibration linkage,
and threshold ordering before readiness succeeds.

The image uses the official `python:3.14.6-slim-bookworm` tag, installs the
locked runtime, runs as an unprivileged user, disables access logs, and includes
the same `/v1/health` readiness probe used by local operations. Compose also
drops Linux capabilities, enables `no-new-privileges`, makes the root filesystem
read-only, and provides only a bounded temporary filesystem.

With the verified local artifacts present, the intended workflow is:

```powershell
docker compose build
docker compose up
```

Then open `http://127.0.0.1:8000` or query
`http://127.0.0.1:8000/v1/health`. Set `AUTHENTITEXT_PORT` before starting
Compose to use another host port.

## Validation boundary

Docker is not installed on the audited workstation, so no local image or
Compose smoke result exists. The repository validates the Dockerfile, Compose
security settings, ignored build context, configurable application root, API,
and artifact readiness behavior with automated tests. CI is configured to run
the real image build after the change reaches GitHub; that hosted result must be
observed before Phase 20 can be marked complete.

The image is a local research service, not a production deployment. It has no
TLS termination, external secrets manager, persistent database, remote
telemetry, autoscaling, or rollback mechanism.
