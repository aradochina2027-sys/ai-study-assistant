
"""
AI Study Assistant — основний сервіс.

ПР15:
    Обробка запитань та валідація відповідей.

ПР16:
    Семантичний пошук RAG.

ПР17:
    Фонові завдання та збереження в SQLite.

ПР18:
    Загальний дедлайн;
    вимикач AI_ENABLED;
    безпечний формат помилок;
    збереження Retry-After.
"""

from __future__ import annotations

import uuid

from math import isfinite

from app.deadlines import (
    check_deadline,
    request_deadline,
)
from app.models import (
    AppError,
    AskInput,
    ModelOutput,
)
from app.orchestration import Orchestrator
from app.rag.embedder import EmbeddingError
from app.rag.index import build_index
from app.rag.retriever import search_index
from app.storage import SQLiteStorage
from app.telemetry import Telemetry


# Повідомлення обираються із заданого списку.
# Текст винятку провайдера не переноситься
# безпосередньо в публічну HTTP-відповідь.

SAFE_ERROR_MESSAGES = {
    "ai_disabled": (
        "AI-функцію тимчасово вимкнено. "
        "Спробуйте пізніше."
    ),
    "timeout": (
        "Модель не відповіла вчасно."
    ),
    "deadline_exceeded": (
        "Перевищено загальний час "
        "обробки запитання."
    ),
    "provider_unavailable": (
        "Провайдер тимчасово недоступний"
    ),
    "rate_limit": (
        "AI-провайдер тимчасово обмежив "
        "кількість запитів."
    ),
    "client_rate_limit": (
        "Надіслано забагато запитів. "
        "Спробуйте пізніше."
    ),
    "concurrency_limit": (
        "Усі місця для AI-запитів зайняті. "
        "Спробуйте пізніше."
    ),
    "circuit_open": (
        "AI-провайдер тимчасово недоступний. "
        "Спробуйте пізніше."
    ),
    "rag_unavailable": (
        "Семантичний пошук тимчасово "
        "недоступний."
    ),
    "rag_document_unsupported": (
        "Вибрані документи не підтримуються "
        "в режимі RAG."
    ),
    "provider_invalid_response": (
        "AI-провайдер повернув некоректну "
        "відповідь."
    ),
    "provider_error": (
        "Не вдалося отримати відповідь "
        "від AI-провайдера."
    ),
    "invalid_output": (
        "Відповідь моделі не пройшла перевірку."
    ),
    "auth": (
        "Не вдалося авторизуватися "
        "в AI-провайдера."
    ),
    "bad_request": (
        "AI-провайдер відхилив запит."
    ),
    "budget": (
        "Вичерпано денний ліміт "
        "викликів AI."
    ),
    "empty_documents": (
        "Вибрані документи порожні."
    ),
    "invalid_document": (
        "Некоректний тип або шлях документа."
    ),
    "document_missing": (
        "Навчальний документ не знайдено."
    ),
    "document_too_large": (
        "Навчальний документ завеликий."
    ),
    "mid_stream": (
        "З'єднання з моделлю перервано."
    ),
}


class StudyService:
    def __init__(
        self,
        storage: SQLiteStorage,
        orchestrator: Orchestrator,
        telemetry: Telemetry,
    ):
        self.storage = storage
        self.orchestrator = orchestrator
        self.telemetry = telemetry

    # ==========================================
    # 1. Перевірка AI_ENABLED
    # ==========================================

    def _check_ai_enabled(
        self,
        request_id: str,
    ) -> None:
        settings = self.orchestrator.settings

        if not getattr(
            settings,
            "ai_enabled",
            True,
        ):
            self.telemetry.emit(
                request_id,
                "service",
                "ai_disabled",
            )

            raise AppError(
                "ai_disabled",
                (
                    "AI-функцію тимчасово "
                    "вимкнено."
                ),
                503,
                False,
            )

    # ==========================================
    # 2. Підготовка RAG-контексту
    # ==========================================

    def _get_rag_context(
        self,
        question: str,
        documents: list[str],
        request_id: str,
    ) -> str | None:
        """
        Повертає:
            None — RAG вимкнено;
            "" — не знайдено фрагментів;
            str — контекст із фрагментів.
        """

        settings = self.orchestrator.settings

        if not settings.rag_enabled:
            return None

        if documents != ["data/notes.md"]:
            raise AppError(
                "rag_document_unsupported",
                (
                    "Семантичний пошук наразі "
                    "підтримує data/notes.md."
                ),
                400,
                False,
            )

        check_deadline()

        self.telemetry.emit(
            request_id,
            "rag",
            "search_started",
        )

        try:
            build_index(
                document_path="data/notes.md",
                index_path=settings.rag_index_path,
                model=settings.rag_embedding_model,
                max_chars=220,
                overlap_chars=40,
            )

            check_deadline()

            self.telemetry.emit(
                request_id,
                "rag",
                "index_ready",
            )

            hits = search_index(
                question,
                index_path=settings.rag_index_path,
                top_k=settings.rag_top_k,
                min_score=settings.rag_min_score,
            )

            check_deadline()

        except (
            EmbeddingError,
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:
            # Якщо вичерпано дедлайн,
            # повідомляємо саме про дедлайн.
            check_deadline()

            self.telemetry.emit(
                request_id,
                "rag",
                "search_unavailable",
            )

            raise AppError(
                "rag_unavailable",
                (
                    "Семантичний пошук тимчасово "
                    "недоступний."
                ),
                503,
                True,
            ) from exc

        self.telemetry.emit(
            request_id,
            "rag",
            f"hits_{len(hits)}",
        )

        if not hits:
            return ""

        blocks: list[str] = []
        used_chars = 0

        for hit in hits:
            check_deadline()

            block = (
                f"<<<DOC:{hit.source}>>>\n"
                f"{hit.text}\n"
                f"<<<END_DOC>>>"
            )

            separator_chars = 2 if blocks else 0

            required_chars = (
                len(block) + separator_chars
            )

            if (
                used_chars + required_chars
                > settings.context_max_chars
            ):
                continue

            blocks.append(block)
            used_chars += required_chars

        if not blocks:
            return ""

        context = "\n\n".join(blocks)

        self.telemetry.emit(
            request_id,
            "rag",
            "context_ready",
        )

        return context

    # ==========================================
    # 3. Основний сценарій
    # ==========================================

    def handle_question(
        self,
        data: AskInput,
        request_id: str | None = None,
    ) -> ModelOutput:
        """
        Повний шлях обробки запитання.

        Спочатку перевіряємо вимикач AI.

        Якщо він вимкнений, не запускаємо:
        - читання контексту;
        - побудову RAG-індексу;
        - embeddings;
        - основну AI-модель.
        """

        request_id = request_id or str(
            uuid.uuid4()
        )

        self._check_ai_enabled(request_id)

        settings = self.orchestrator.settings

        total_timeout_s = getattr(
            settings,
            "request_timeout_s",
            None,
        )

        if total_timeout_s is None:
            total_timeout_s = (
                settings.llm_timeout_s
            )

        with request_deadline(
            total_timeout_s
        ):
            return self._handle_with_deadline(
                data,
                request_id,
            )

    # ==========================================
    # 4. Виконання в межах дедлайну
    # ==========================================

    def _handle_with_deadline(
        self,
        data: AskInput,
        request_id: str,
    ) -> ModelOutput:
        self.telemetry.emit(
            request_id,
            "service",
            "started",
        )

        check_deadline()

        # --------------------------------------
        # 4.1. Завантаження документів
        # --------------------------------------

        original_context = (
            self.storage.load_documents(
                data.documents,
                request_id,
            )
        )

        check_deadline()

        # --------------------------------------
        # 4.2. Додатковий компонент RAG
        # --------------------------------------

        rag_context = self._get_rag_context(
            data.question,
            data.documents,
            request_id,
        )

        check_deadline()

        # --------------------------------------
        # 4.3. Вибір контексту
        # --------------------------------------

        if rag_context is None:
            context = original_context

        elif not rag_context:
            result = ModelOutput(
                status="no_evidence",
                answer=(
                    "У матеріалах немає "
                    "підтвердженої відповіді."
                ),
                sources=[],
            )

            self.telemetry.emit(
                request_id,
                "service",
                "no_evidence",
            )

            self.telemetry.emit(
                request_id,
                "service",
                "finished",
            )

            return result

        else:
            context = rag_context

        check_deadline()

        # --------------------------------------
        # 4.4. Генерація та перевірка
        # --------------------------------------

        result = self.orchestrator.answer(
            data.question,
            data.documents,
            context,
            request_id,
        )

        check_deadline()

        self.telemetry.emit(
            request_id,
            "service",
            "finished",
        )

        return result

    # ==========================================
    # 5. Безпечне збереження помилок
    # ==========================================

    @staticmethod
    def _error_payload(
        error: AppError,
    ) -> dict:
        """
        Перетворює AppError на дані для SQLite.

        Сторонні тексти винятків,
        API-ключі та traceback не зберігаємо.

        Retry-After зберігається тільки
        для дозволених контрольованих помилок.
        """

        safe_message = SAFE_ERROR_MESSAGES.get(
            error.kind
        )

        if safe_message is None:
            return {
                "kind": "internal_error",
                "message": (
                    "Внутрішня помилка сервера."
                ),
                "retryable": False,
                "status_code": 500,
            }

        payload = {
            "kind": error.kind,
            "message": safe_message,
            "retryable": bool(error.retryable),
            "status_code": error.status_code,
        }

        retry_after = getattr(
            error,
            "retry_after_s",
            None,
        )

        if (
            error.retryable
            and error.status_code in {429, 503, 504}
            and retry_after is not None
            and not isinstance(retry_after, bool)
        ):
            try:
                seconds = float(retry_after)
            except (TypeError, ValueError):
                seconds = 0.0

            if (
                isfinite(seconds)
                and seconds > 0
            ):
                payload["retry_after_s"] = seconds

        return payload

    # ==========================================
    # 6. Фонове виконання FastAPI
    # ==========================================

    def run_job(
        self,
        data: AskInput,
        request_id: str,
    ) -> None:
        """
        Результат або контрольована помилка
        зберігаються у SQLite.

        Серверний GET /api/requests/{request_id}
        читає збережені дані та формує
        HTTP-відповідь для браузера.
        """

        self.storage.set_running(
            request_id
        )

        try:
            result = self.handle_question(
                data,
                request_id,
            )

            self.storage.complete(
                request_id,
                result.model_dump(),
            )

        except AppError as error:
            self.telemetry.emit(
                request_id,
                "service",
                f"error_{error.kind}",
            )

            self.storage.fail(
                request_id,
                self._error_payload(error),
            )

        except Exception:
            # Не виводимо repr(exception),
            # traceback, промпт чи API-ключ.

            self.telemetry.emit(
                request_id,
                "service",
                "internal_error",
            )

            self.storage.fail(
                request_id,
                {
                    "kind": "internal_error",
                    "message": (
                        "Внутрішня помилка сервера."
                    ),
                    "retryable": False,
                    "status_code": 500,
                },
            )
