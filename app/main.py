import json
import time
import uuid

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.gateway import Gateway
from app.models import (
    AppError,
    AskInput,
    ErrorDetail,
    ErrorResponse,
    JobAccepted,
    JobView,
    ModelOutput,
)
from app.orchestration import Orchestrator
from app.providers.mock import MockProvider
from app.providers.unavailable import NotImplementedProvider
from app.service import StudyService
from app.settings import Settings, load_settings
from app.storage import SQLiteStorage
from app.telemetry import Telemetry


def create_app(
    settings: Settings | None = None,
) -> FastAPI:

    settings = settings or load_settings()
    settings.validate()

    telemetry = Telemetry(settings.log_file)
    storage = SQLiteStorage(
        settings.db_path,
        telemetry,
    )

    if settings.llm_provider == "mock":
        provider = MockProvider(
            settings, telemetry
        )
    else:
        provider = NotImplementedProvider()

    gateway = Gateway(provider, telemetry)

    orchestrator = Orchestrator(
        gateway, settings, telemetry
    )

    service = StudyService(
        storage, orchestrator, telemetry
    )

    api = FastAPI(
        title="AI Study Assistant",
        version=settings.app_version,
    )

    api.state.service = service
    api.state.storage = storage

    started_at = time.monotonic()

    def error_response(
        err: AppError,
        request_id: str,
    ) -> JSONResponse:

        payload = ErrorResponse(
            error=ErrorDetail(
                kind=err.kind,
                message=err.message,
                retryable=err.retryable,
                request_id=request_id,
            )
        )

        return JSONResponse(
            status_code=err.status_code,
            content=payload.model_dump(),
        )

    @api.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        return error_response(
            AppError(
                "validation_error",
                "Перевірте формат і довжину полів",
                422,
            ),
            request.headers.get(
                "X-Request-ID",
                str(uuid.uuid4()),
            ),
        )

    @api.get("/")
    def index():
        return FileResponse(
            Path(__file__).resolve().parent.parent
            / "static"
            / "index.html"
        )

    @api.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": settings.app_version,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "mode": (
                "mock"
                if settings.llm_provider == "mock"
                else "planned"
            ),
            "storage": (
                "ok"
                if storage.ready()
                else "unavailable"
            ),
            "extra": {
                "index": "planned_pr16"
            },
            "uptime_s": round(
                time.monotonic() - started_at,
                2,
            ),
        }

    @api.post(
        "/api/ask",
        response_model=JobAccepted,
        status_code=202,
    )
    def ask(
        body: AskInput,
        background_tasks: BackgroundTasks,
    ):
        request_id = str(uuid.uuid4())

        telemetry.emit(
            request_id,
            "controller",
            "accepted",
        )

        storage.create(request_id)

        background_tasks.add_task(
            service.run_job,
            body,
            request_id,
        )

        return JobAccepted(
            request_id=request_id,
            status_url=(
                f"/api/requests/{request_id}"
            ),
        )

    @api.get(
        "/api/requests/{request_id}",
        response_model=JobView,
        responses={
            404: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def get_request(request_id: str):

        telemetry.emit(
            request_id,
            "controller",
            "get_result",
        )

        record = storage.get(request_id)

        if record is None:
            return error_response(
                AppError(
                    "not_found",
                    "Запит не знайдено",
                    404,
                ),
                request_id,
            )

        if record.status == "failed":
            err = json.loads(
                record.error_json or "{}"
            )

            return error_response(
                AppError(
                    err.get("kind", "failed"),
                    err.get("message", "Помилка"),
                    err.get("status_code", 503),
                    err.get("retryable", False),
                ),
                request_id,
            )

        result = (
            ModelOutput.model_validate_json(
                record.result_json
            )
            if record.result_json
            else None
        )

        return JobView(
            request_id=request_id,
            status=record.status,
            result=result,
        )

    @api.post(
        "/api/requests/{request_id}/cancel"
    )
    def cancel_request(request_id: str):

        telemetry.emit(
            request_id,
            "controller",
            "cancel",
        )

        record = storage.get(request_id)

        if record is None:
            return error_response(
                AppError(
                    "not_found",
                    "Запит не знайдено",
                    404,
                ),
                request_id,
            )

        if not storage.cancel(request_id):
            return error_response(
                AppError(
                    "conflict",
                    "Запит уже завершився",
                    409,
                ),
                request_id,
            )

        return {
            "request_id": request_id,
            "status": "cancelled",
        }

    return api


app = create_app()