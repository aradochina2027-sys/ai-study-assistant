
"""
AI Study Assistant — FastAPI.

ПР14:
    API, SQLite, фонові завдання.

ПР15:
    AI-провайдер, Gateway, бюджет.

ПР16:
    RAG через StudyService.

ПР17:
    Вебінтерфейс, валідація,
    Idempotency-Key, відновлення запитів.

ПР18:
    Власний ліміт запитів;
    Retry-After;
    стан Circuit Breaker у health.
"""

import hashlib
import json
import math
import os
import re
import time
import uuid

from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    Request,
)
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
    """
    Створює окремий екземпляр FastAPI.

    Компоненти створюються окремо
    для кожного екземпляра застосунку,
    що також дозволяє ізолювати тести.
    """

    # ==========================================
    # 1. Конфігурація
    # ==========================================

    if settings is None:
        settings = load_settings()

    settings.validate()

    telemetry = Telemetry(
        settings.log_file
    )

    # ==========================================
    # 3. SQLite
    # ==========================================

    storage = SQLiteStorage(
        settings.db_path,
        telemetry,
    )

    # ==========================================
    # 4. AI-провайдер
    # ==========================================

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

    # ==========================================
    # 5. Денний бюджет
    # ==========================================

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

    # ==========================================
    # 6. Gateway із Circuit Breaker
    # ==========================================

    gateway = Gateway(
        provider,
        telemetry,
        budget,
    )

    # ==========================================
    # 7. Orchestrator
    # ==========================================

    orchestrator = Orchestrator(
        gateway,
        settings,
        telemetry,
    )

    # ==========================================
    # 8. StudyService
    # ==========================================

    service = StudyService(
        storage,
        orchestrator,
        telemetry,
    )

    # ==========================================
    # 9. Власний ліміт частоти
    # ==========================================

    api = FastAPI(
        title="AI Study Assistant",
        description=(
            "Навчальний AI-помічник із "
            "локальною AI-моделлю, RAG "
            "та захистом від збоїв."
        ),
        version=settings.app_version,
    )

    # Компоненти для автоматизованих тестів.
    api.state.service = service
    api.state.storage = storage
    api.state.orchestrator = orchestrator
    api.state.gateway = gateway
    api.state.telemetry = telemetry

    started_at = time.monotonic()

    # ==========================================
    # 11. Формат контрольованої помилки
    # ==========================================

    def error_response(
        err: AppError,
        request_id: str,
        fields: list[dict[str, str]] | None = None,
    ) -> JSONResponse:
        """
        Єдиний формат помилок ПР14–ПР18.

        Підтримує:
        - вид помилки;
        - повідомлення;
        - retryable;
        - request_id;
        - помилки конкретних полів;
        - заголовок Retry-After.
        """

        detail = None

        if fields is not None:
            detail = [
                {
                    "loc": [
                        "body",
                        item["field"],
                    ],
                    "msg": item["message"],
                }
                for item in fields
            ]

        payload = ErrorResponse(
            error=ErrorDetail(
                kind=err.kind,
                message=err.message,
                retryable=err.retryable,
                request_id=request_id,
                fields=fields,
            ),
            detail=detail,
        )

        headers = {
            "X-Request-ID": request_id,
            "Cache-Control": "no-store",
        }

        # ПР18 — Retry-After у HTTP-заголовку.
        telemetry.emit(
            request_id,
            "controller",
            f"error_{err.kind}",
        )

        return JSONResponse(
            status_code=err.status_code,
            content=payload.model_dump(
                exclude_none=True
            ),
            headers=headers,
        )

    # ==========================================
    # 12. Помилки валідації
    # ==========================================

    @api.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        request_id = str(uuid.uuid4())

        field_errors: list[dict[str, str]] = []

        for issue in exc.errors():
            location = issue.get(
                "loc",
                [],
            )

            field = next(
                (
                    item
                    for item in reversed(location)
                    if isinstance(item, str)
                    and item not in {
                        "body",
                        "query",
                        "path",
                        "header",
                    }
                ),
                "request",
            )

            if field == "question":
                message = (
                    "Запитання має містити "
                    "від 1 до 500 символів."
                )

            elif field == "documents":
                message = (
                    "Потрібно вибрати "
                    "від 1 до 5 документів."
                )

            else:
                message = (
                    "Перевірте правильність "
                    "значення цього поля."
                )

            field_errors.append(
                {
                    "field": field,
                    "message": message,
                }
            )

        return error_response(
            AppError(
                "validation_error",
                "Перевірте введені дані.",
                422,
                False,
            ),
            request_id,
            fields=field_errors,
        )

    # ==========================================
    # 13. Обробка AppError
    # ==========================================

    @api.exception_handler(AppError)
    async def app_error_handler(
        request: Request,
        exc: AppError,
    ):
        request_id = str(uuid.uuid4())

        return error_response(
            exc,
            request_id,
        )

    # ==========================================
    # 14. Головна сторінка StudyAI
    # ==========================================

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
            headers={
                "Cache-Control": (
                    "no-store, no-cache, "
                    "must-revalidate, max-age=0"
                ),
                "Pragma": "no-cache",
                "Expires": "0",
                "X-UI-Version": "StudyAI-PR17",
            },
        )

    # ==========================================
    # 15. GET /api/health
    # ==========================================

    @api.get("/api/health")
    def health():
        """
        Показує стан системи.

        Не викликає модель під час
        перевірки стану Circuit Breaker.
        """

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

        rag_enabled = bool(getattr(settings, "rag_enabled", False))

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
                "rag": (
                    "enabled"
                    if rag_enabled
                    else "disabled"
                ),
            },
            "uptime_s": round(
                time.monotonic() - started_at,
                2,
            ),
        }

    # ==========================================
    # 16. POST /api/ask
    # ==========================================

    @api.post(
        "/api/ask",
        response_model=JobAccepted,
        status_code=202,
        responses={
            400: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def ask(
        body: AskInput,
        background_tasks: BackgroundTasks,
        request: Request,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
        ),
    ):
        """
        Створює запит до AI.

        ПР17:
            Idempotency-Key захищає
            від дублювання завдань.

        ПР18:
            Частота запитів перевіряється
            перед записом до SQLite.
        """

        request_id = str(uuid.uuid4())

        # --------------------------------------
        # 16.1. Перевірка Idempotency-Key
        # --------------------------------------

        key = None

        if idempotency_key is not None:
            key = idempotency_key.strip()

            if not re.fullmatch(
                r"[A-Za-z0-9._:-]{8,128}",
                key,
            ):
                return error_response(
                    AppError(
                        "invalid_idempotency_key",
                        (
                            "Idempotency-Key має "
                            "містити 8–128 символів: "
                            "літери, цифри, крапку, "
                            "дефіс, двокрапку або _."
                        ),
                        400,
                        False,
                    ),
                    request_id,
                )

        # --------------------------------------
        # 16.2. Власний ліміт запитів
        # --------------------------------------

        # Використовуємо адресу з безпосереднього
        # HTTP-з'єднання, а не довіряємо
        # довільному X-Forwarded-For.
        if key is not None:
            # Хешуємо канонічний вміст запиту.
            payload = json.dumps(
                body.model_dump(
                    mode="json"
                ),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            payload_hash = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

            try:
                request_id, created = (
                    storage.create_idempotent(
                        request_id,
                        key,
                        payload_hash,
                    )
                )

            except AppError as err:
                return error_response(
                    err,
                    request_id,
                )

            if not created:
                telemetry.emit(
                    request_id,
                    "controller",
                    "idempotency_replayed",
                )

                # Не створюємо другу фонову задачу.
                return JobAccepted(
                    request_id=request_id,
                    status="pending",
                    status_url=(
                        f"/api/requests/{request_id}"
                    ),
                )

        else:
            # Сумісність зі старими клієнтами,
            # які не передають ключ.
            storage.create(
                request_id
            )

        # --------------------------------------
        # 16.4. Запуск нової фонової задачі
        # --------------------------------------

        telemetry.emit(
            request_id,
            "controller",
            "accepted",
        )

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

    # ==========================================
    # 17. GET /api/requests/{request_id}
    # ==========================================

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
    def get_request(
        request_id: str,
    ):
        telemetry.emit(
            request_id,
            "controller",
            "get_result",
        )

        record = storage.get(
            request_id
        )

        if record is None:
            return error_response(
                AppError(
                    "not_found",
                    "Запит не знайдено",
                    404,
                ),
                request_id,
            )

        # --------------------------------------
        # 17.1. Контрольована помилка завдання
        # --------------------------------------

        if record.status == "failed":
            try:
                err_data = json.loads(
                    record.error_json or "{}"
                )
            except ValueError:
                err_data = {}

            if not isinstance(err_data, dict):
                err_data = {}

            err = AppError(
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
            )

            # Якщо сервіс зберіг Retry-After,
            # передаємо його клієнту.
            return error_response(
                err,
                request_id,
            )

        # --------------------------------------
        # 17.2. Успішна відповідь
        # --------------------------------------

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

    # ==========================================
    # 18. Скасування запиту
    # ==========================================

    @api.post(
        "/api/requests/{request_id}/cancel",
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    def cancel_request(
        request_id: str,
    ):
        telemetry.emit(
            request_id,
            "controller",
            "cancel_requested",
        )

        record = storage.get(
            request_id
        )

        if record is None:
            return error_response(
                AppError(
                    "not_found",
                    "Запит не знайдено",
                    404,
                ),
                request_id,
            )

        if not storage.cancel(
            request_id
        ):
            return error_response(
                AppError(
                    "conflict",
                    (
                        "Запит уже завершено "
                        "або скасовано"
                    ),
                    409,
                ),
                request_id,
            )

        return {
            "request_id": request_id,
            "status": "cancelled",
        }

    return api


# Об'єкт для Uvicorn.
app = create_app()
