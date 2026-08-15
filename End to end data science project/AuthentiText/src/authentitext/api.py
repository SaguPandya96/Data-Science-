"""FastAPI service for the frozen AuthentiText inference contract."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from authentitext import __version__
from authentitext.drift import DriftError, evaluate_drift, load_drift_reference
from authentitext.inference import AuthentiTextPredictor, PredictionError
from authentitext.monitoring import OBSERVED_ENDPOINTS, OperationalMetrics

API_VERSION = "v1"
MAX_BATCH_ITEMS = 32
MAX_BATCH_CHARACTERS = 200_000
LOGGER = logging.getLogger("authentitext.api")
WEB_ROOT = Path(__file__).resolve().parent / "web"


def application_root() -> Path:
    """Resolve runtime metadata and artifact paths for source or installed builds."""
    configured = os.environ.get("AUTHENTITEXT_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[2]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


def _error_response(status_code: int, code: str, message: str, **details: Any) -> JSONResponse:
    error = {"code": code, "message": message, **details}
    return JSONResponse(status_code=status_code, content={"error": error})


def _default_predictor() -> AuthentiTextPredictor:
    repo_root = application_root()
    return AuthentiTextPredictor.from_reports(
        training_report_path=(
            repo_root / "data" / "metadata" / "mage_baseline_training_report.json"
        ),
        calibration_report_path=repo_root / "data" / "metadata" / "mage_calibration_report.json",
        artifact_root=repo_root / "artifacts" / "baselines" / "id",
    )


def _default_drift_reference() -> dict[str, Any]:
    repo_root = application_root()
    return load_drift_reference(repo_root / "data" / "metadata" / "mage_drift_reference.json")


def create_app(
    *,
    predictor: AuthentiTextPredictor | Any | None = None,
    load_default_predictor: bool = True,
    metrics: OperationalMetrics | None = None,
    drift_reference: dict[str, Any] | None = None,
    load_default_drift_reference: bool = True,
) -> FastAPI:
    """Build an application with injectable model state for testing."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.predictor = predictor
        application.state.load_error = None
        application.state.drift_reference = drift_reference
        application.state.drift_load_error = None
        if application.state.predictor is None and load_default_predictor:
            try:
                application.state.predictor = _default_predictor()
            except PredictionError as error:
                application.state.load_error = error.code
                LOGGER.error("model readiness failed code=%s", error.code)
        if application.state.drift_reference is None and load_default_drift_reference:
            try:
                application.state.drift_reference = _default_drift_reference()
            except DriftError:
                application.state.drift_load_error = "drift_reference_invalid"
                LOGGER.error("drift readiness failed code=drift_reference_invalid")
        yield
        application.state.predictor = None
        application.state.drift_reference = None

    application = FastAPI(
        title="AuthentiText API",
        summary="Calibrated, abstaining machine-generated text research baseline",
        version=__version__,
        lifespan=lifespan,
    )
    metric_store = metrics or OperationalMetrics()
    application.state.metrics = metric_store
    application.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @application.middleware("http")
    async def prediction_request_metrics(request: Request, call_next: Any) -> Any:
        endpoint = request.url.path
        if request.method != "POST" or endpoint not in OBSERVED_ENDPOINTS:
            return await call_next(request)
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metric_store.record_http_request(endpoint, 500, (perf_counter() - started) * 1_000)
            metric_store.record_error("unhandled_exception")
            raise
        metric_store.record_http_request(
            endpoint, response.status_code, (perf_counter() - started) * 1_000
        )
        return response

    @application.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "script-src 'self'; style-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        if request.url.path in OBSERVED_ENDPOINTS:
            metric_store.record_error("request_validation")
        issues = [
            {
                "location": [str(part) for part in issue.get("loc", ())],
                "type": issue.get("type", "validation_error"),
                "message": issue.get("msg", "Request validation failed"),
            }
            for issue in error.errors()
        ]
        return _error_response(
            422,
            "request_validation",
            "Request body does not match the API contract",
            issues=issues,
        )

    def ready_predictor() -> AuthentiTextPredictor | Any:
        loaded = application.state.predictor
        if loaded is None:
            raise PredictionError("model_not_ready", "The verified model is not ready")
        return loaded

    @application.get("/", include_in_schema=False, response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @application.get("/health/live", tags=["health"])
    async def live() -> dict[str, Any]:
        return {"status": "ok", "service": "authentitext", "version": __version__}

    @application.get("/health/ready", tags=["health"], response_model=None)
    async def ready() -> dict[str, Any] | JSONResponse:
        loaded = application.state.predictor
        if loaded is None:
            return _error_response(503, "model_not_ready", "The verified model is not ready")
        return {
            "status": "ready",
            "model": "word_tfidf_logistic_isotonic",
            "base_model_sha256": loaded.identity.base_model_sha256,
            "calibration_sha256": loaded.identity.calibration_sha256,
        }

    @application.get(f"/{API_VERSION}/version", tags=["metadata"])
    async def version() -> dict[str, Any]:
        return {
            "service": "authentitext",
            "service_version": __version__,
            "api_version": API_VERSION,
            "inference_schema_version": 1,
        }

    @application.get(f"/{API_VERSION}/model", tags=["metadata"], response_model=None)
    async def model() -> dict[str, Any] | JSONResponse:
        try:
            loaded = ready_predictor()
        except PredictionError as error:
            return _error_response(503, error.code, str(error))
        return {
            "name": "word_tfidf_logistic_isotonic",
            "dataset_id": loaded.identity.dataset_id,
            "dataset_revision": loaded.identity.revision,
            "base_model_sha256": loaded.identity.base_model_sha256,
            "calibration_sha256": loaded.identity.calibration_sha256,
            "thresholds": {
                "likely_human_max": round(loaded.human_threshold, 12),
                "likely_machine_min": round(loaded.machine_threshold, 12),
            },
        }

    @application.get(f"/{API_VERSION}/metrics", tags=["operations"], response_model=None)
    async def metrics_snapshot() -> dict[str, Any]:
        snapshot = metric_store.snapshot()
        snapshot["service"] = {"name": "authentitext", "version": __version__}
        loaded = application.state.predictor
        snapshot["model"] = (
            {
                "ready": True,
                "name": "word_tfidf_logistic_isotonic",
                "base_model_sha256": loaded.identity.base_model_sha256,
                "calibration_sha256": loaded.identity.calibration_sha256,
            }
            if loaded is not None
            else {"ready": False}
        )
        return snapshot

    @application.get(f"/{API_VERSION}/drift", tags=["operations"], response_model=None)
    async def drift_snapshot() -> dict[str, Any] | JSONResponse:
        reference = application.state.drift_reference
        loaded = application.state.predictor
        if reference is None:
            return _error_response(
                503, "drift_reference_not_ready", "The verified drift reference is not ready"
            )
        if loaded is None:
            return _error_response(503, "model_not_ready", "The verified model is not ready")
        model = reference["identity"]["model"]
        if (
            model["base_model_sha256"] != loaded.identity.base_model_sha256
            or model["calibration_sha256"] != loaded.identity.calibration_sha256
        ):
            return _error_response(
                409, "drift_model_mismatch", "The drift reference belongs to another model"
            )
        try:
            return evaluate_drift(metric_store.snapshot(), reference)
        except DriftError:
            LOGGER.error("drift evaluation failed code=drift_evaluation_invalid")
            return _error_response(
                503, "drift_evaluation_invalid", "Aggregate drift evaluation is unavailable"
            )

    @application.post(f"/{API_VERSION}/predict", tags=["prediction"], response_model=None)
    async def predict(request: PredictRequest) -> dict[str, Any] | JSONResponse:
        try:
            loaded = ready_predictor()
            result = loaded.predict(request.text)
        except PredictionError as error:
            metric_store.record_error(error.code)
            status_code = 503 if error.code == "model_not_ready" else 422
            return _error_response(status_code, error.code, str(error))
        metric_store.record_predictions((result,))
        LOGGER.info(
            "prediction completed category=%s characters=%s tokens=%s",
            result["category"],
            result["input_summary"]["characters"],
            result["input_summary"]["whitespace_tokens"],
        )
        return result

    @application.post(f"/{API_VERSION}/predict/batch", tags=["prediction"], response_model=None)
    async def predict_batch(request: BatchPredictRequest) -> dict[str, Any] | JSONResponse:
        if sum(len(text) for text in request.texts) > MAX_BATCH_CHARACTERS:
            metric_store.record_error("batch_too_large")
            return _error_response(
                422,
                "batch_too_large",
                f"Batch must contain at most {MAX_BATCH_CHARACTERS} characters in total",
            )
        try:
            loaded = ready_predictor()
        except PredictionError as error:
            metric_store.record_error(error.code)
            return _error_response(503, error.code, str(error))
        results = []
        for index, text in enumerate(request.texts):
            try:
                results.append(loaded.predict(text))
            except PredictionError as error:
                metric_store.record_error("batch_item_invalid")
                return _error_response(
                    422,
                    "batch_item_invalid",
                    str(error),
                    item_index=index,
                    item_code=error.code,
                )
        metric_store.record_predictions(results)
        LOGGER.info("batch prediction completed items=%s", len(results))
        return {"schema_version": 1, "count": len(results), "results": results}

    return application


app = create_app()
