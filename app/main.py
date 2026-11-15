
import json
import time
import uuid

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.budget import DailyBudget
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
from app.providers.openai_compat import OpenAICompatibleProvider
from app.providers.recorded import RecordedProvider
from app.service import StudyService
from app.settings import Settings, load_settings
from app.storage import SQLiteStorage
from app.telemetry import Telemetry


def create_app(
    settings: Settings | None = None,
) -> FastAPI:

    # 1. Завантаження конфігурації
    settings = settings or load_settings()
    settings.validate()

    # 2. Журналювання
    telemetry = Telemetry(settings.log_file)

    # 3. Сховище SQLite
    storage = SQLiteStorage(
        settings.db_path,
        telemetry,
    )

    # 4. Вибір AI-провайдера
    if settings.llm_provider == "mock":
        provider = MockProvider(
            settings,
            telemetry,
        )

    elif settings.llm_provider in {
        "ollama",
        "openai_compat",
    }:
        provider = OpenAICompatibleProvider(
            settings
        )

    elif settings.llm_provider == "recorded":

        upstream = None

        if settings.record_mode != "replay":
            upstream = OpenAICompatibleProvider(
                settings
            )

        provider = RecordedProvider(
            settings,
            upstream,
        )

    else:
        raise ValueError(
            "Unknown AI provider: "
            + settings.llm_provider
        )

    # 5. Денний ліміт AI-викликів
    budget = None

    if settings.llm_provider in {
        "ollama",
        "openai_compat",
    } or (
        settings.llm_provider == "recorded"
        and settings.record_mode != "replay"
    ):
        budget = DailyBudget(
            settings.db_path,
            settings.daily_call_limit,
        )

    # 6. Шлюз провайдера
    gateway = Gateway(
        provider,
        telemetry,
        budget,
    )

    # 7. Оркестрація AI
    orchestrator = Orchestrator(
        gateway,
        settings,
        telemetry,
    )

    # 8. Прикладний сервіс
    service = StudyService(
        storage,
        orchestrator,
        telemetry,
    )

    # 9. Створення FastAPI
    api = FastAPI(
        title="AI Study Assistant",
        description=(
            "Навчальний AI-помічник. "
            "Практична робота №15."
        ),
        version=settings.app_version,
    )

    # Компоненти для тестів та eval
    api.state.service = service
    api.state.storage = storage
    api.state.orchestrator = orchestrator
    api.state.gateway = gateway
    api.state.telemetry = telemetry

    started_at = time.monotonic()

    # Єдиний формат помилок
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

        telemetry.emit(
            request_id,
            "controller",
            f"error_{err.kind}",
        )

        return JSONResponse(
            status_code=err.status_code,
            content=payload.model_dump(),
        )

    # Помилки перевірки введених даних
    @api.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        request_id = str(uuid.uuid4())

        return error_response(
            AppError(
                "validation_error",
                "Перевірте формат і довжину полів",
                422,
                False,
            ),
            request_id,
        )

    # ------------------------------------------------
    # Головна сторінка
    # ------------------------------------------------

    @api.get("/")
    def index():
        path = (
            Path(__file__).resolve().parent.parent
            / "static"
            / "index.html"
        )

        return FileResponse(
            path,
            media_type="text/html",
        )

    # ------------------------------------------------
    # GET /api/health
    # ------------------------------------------------

    @api.get("/api/health")
    def health():

        if settings.llm_provider == "mock":
            mode = "mock"

        elif settings.llm_provider == "recorded":
            mode = settings.record_mode

        else:
            mode = "real"

        storage_status = (
            "ok"
            if storage.ready()
            else "unavailable"
        )

        return {
            "status": (
                "ok"
                if storage_status == "ok"
                else "degraded"
            ),
            "version": settings.app_version,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "mode": mode,
            "storage": storage_status,
            "extra": {
                "index": "planned_pr16"
            },
            "uptime_s": round(
                time.monotonic() - started_at,
                2,
            ),
        }

    # ------------------------------------------------
    # POST /api/ask
    # ------------------------------------------------

    @api.post(
        "/api/ask",
        response_model=JobAccepted,
        status_code=202,
        responses={
            422: {"model": ErrorResponse},
        },
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

        # Створюємо запис у сховищі
        storage.create(request_id)

        # Запускаємо фонову обробку
        background_tasks.add_task(
            service.run_job,
            body,
            request_id,
        )

        return JobAccepted(
            request_id=request_id,
            status="pending",
            status_url=(
                f"/api/requests/{request_id}"
            ),
        )

    # ------------------------------------------------
    # GET /api/requests/{request_id}
    # ------------------------------------------------

    @api.get(
        "/api/requests/{request_id}",
        response_model=JobView,
        responses={
            404: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    def get_request(request_id: str):

        telemetry.emit(
            request_id,
            "controller",
            "get_result",
        )

        record = storage.get(request_id)

        # Запит не існує
        if record is None:
            return error_response(
                AppError(
                    "not_found",
                    "Запит не знайдено",
                    404,
                ),
                request_id,
            )

        # Збережена помилка AI
        if record.status == "failed":

            try:
                err_data = json.loads(
                    record.error_json or "{}"
                )
            except ValueError:
                err_data = {}

            return error_response(
                AppError(
                    kind=err_data.get(
                        "kind",
                        "provider_error",
                    ),
                    message=err_data.get(
                        "message",
                        "Помилка обробки запиту",
                    ),
                    status_code=err_data.get(
                        "status_code",
                        503,
                    ),
                    retryable=err_data.get(
                        "retryable",
                        False,
                    ),
                ),
                request_id,
            )

        # Успішний структурований результат
        result = None

        if record.result_json:
            result = ModelOutput.model_validate_json(
                record.result_json
            )

        return JobView(
            request_id=request_id,
            status=record.status,
            result=result,
        )

    # ------------------------------------------------
    # POST /api/requests/{request_id}/cancel
    # ------------------------------------------------

    @api.post(
        "/api/requests/{request_id}/cancel",
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    def cancel_request(request_id: str):

        telemetry.emit(
            request_id,
            "controller",
            "cancel_requested",
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
                    "Запит уже завершено або скасовано",
                    409,
                ),
                request_id,
            )

        return {
            "request_id": request_id,
            "status": "cancelled",
        }

    return api


# Головний об'єкт FastAPI
app = create_app()
