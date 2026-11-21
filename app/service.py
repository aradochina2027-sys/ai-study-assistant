
from __future__ import annotations

import uuid

from app.models import AppError, AskInput, ModelOutput
from app.orchestration import Orchestrator
from app.rag.embedder import EmbeddingError
from app.rag.index import build_index
from app.rag.retriever import search_index
from app.storage import SQLiteStorage
from app.telemetry import Telemetry


class StudyService:
    """
    Основний сервіс AI Study Assistant.

    ПР15:
    - завантаження документів;
    - генерація відповіді;
    - перевірка результату;
    - збереження результату;
    - контрольовані помилки.

    ПР16:
    - семантичний пошук RAG;
    - побудова векторного індексу;
    - пошук релевантних фрагментів;
    - формування контексту для Qwen3.
    """

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
    # ПР16 — Семантичний пошук RAG
    # ==========================================

    def _get_rag_context(
        self,
        question: str,
        documents: list[str],
        request_id: str,
    ) -> str | None:
        """
        Повертає контекст із релевантних фрагментів.

        None:
            RAG вимкнений.

        Порожній рядок:
            релевантних фрагментів немає.

        Непорожній рядок:
            знайдено контекст для основної моделі.
        """

        settings = self.orchestrator.settings

        # Якщо RAG вимкнено, залишаємо
        # стандартний сценарій ПР15.
        if not settings.rag_enabled:
            return None

        # На поточному етапі підтримується
        # лише один документ.
        if documents != ["data/notes.md"]:
            raise AppError(
                "rag_document_unsupported",
                "Семантичний пошук наразі підтримує "
                "лише data/notes.md.",
                400,
                False,
            )

        self.telemetry.emit(
            request_id,
            "rag",
            "search_started",
        )

        try:
            # ==================================
            # 1. Створення або оновлення індексу
            # ==================================

            # Якщо документ не змінювався,
            # build_index використовує
            # раніше створений індекс.
            build_index(
                document_path="data/notes.md",
                index_path=settings.rag_index_path,
                model=settings.rag_embedding_model,
                max_chars=220,
                overlap_chars=40,
            )

            self.telemetry.emit(
                request_id,
                "rag",
                "index_ready",
            )

            # ==================================
            # 2. Семантичний пошук
            # ==================================

            hits = search_index(
                question,
                index_path=settings.rag_index_path,
                top_k=settings.rag_top_k,
                min_score=settings.rag_min_score,
            )

        except (
            EmbeddingError,
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:

            self.telemetry.emit(
                request_id,
                "rag",
                "search_unavailable",
            )

            raise AppError(
                "rag_unavailable",
                "Семантичний пошук тимчасово "
                "недоступний. Перевірте роботу "
                "Ollama та індекс документів.",
                503,
                True,
            ) from exc

        # ======================================
        # 3. Журналювання кількості результатів
        # ======================================

        self.telemetry.emit(
            request_id,
            "rag",
            f"hits_{len(hits)}",
        )

        # Якщо нічого не знайдено,
        # не передаємо моделі порожній контекст.
        if not hits:
            return ""

        # ======================================
        # 4. Формування контексту
        # ======================================

        blocks: list[str] = []
        used_chars = 0

        for hit in hits:

            # Кожен фрагмент містить
            # назву джерела.
            block = (
                f"<<<DOC:{hit.source}>>>\n"
                f"{hit.text}\n"
                f"<<<END_DOC>>>"
            )

            # Додаємо розділювач між блоками.
            separator_size = 2 if blocks else 0

            needed_chars = (
                len(block) + separator_size
            )

            # Не перевищуємо максимальний
            # розмір контексту з ПР15.
            if (
                used_chars + needed_chars
                > settings.context_max_chars
            ):
                continue

            blocks.append(block)
            used_chars += needed_chars

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
    # ПР15 + ПР16 — Основний AI-сценарій
    # ==========================================

    def handle_question(
        self,
        data: AskInput,
        request_id: str | None = None,
    ) -> ModelOutput:
        """
        Повний сценарій:

        1. Прийняти запитання.
        2. Перевірити документи.
        3. Виконати RAG, якщо ввімкнено.
        4. Сформувати контекст.
        5. Викликати Qwen3 через Orchestrator.
        6. Повернути перевірений результат.
        """

        request_id = request_id or str(
            uuid.uuid4()
        )

        self.telemetry.emit(
            request_id,
            "service",
            "started",
        )

        # ======================================
        # 1. Перевірка та завантаження документів
        # ======================================

        original_context = (
            self.storage.load_documents(
                data.documents,
                request_id,
            )
        )

        # ======================================
        # 2. Додатковий AI-компонент RAG
        # ======================================

        rag_context = self._get_rag_context(
            data.question,
            data.documents,
            request_id,
        )

        # ======================================
        # 3. Вибір контексту
        # ======================================

        if rag_context is None:

            # RAG вимкнено.
            # Використовуємо старий сценарій ПР15.
            context = original_context

        elif not rag_context:

            # RAG працює, але підтвердженої
            # інформації не знайдено.
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

            # Передаємо основній моделі тільки
            # релевантні навчальні фрагменти.
            context = rag_context

        # ======================================
        # 4. Генерація відповіді через Qwen3
        # ======================================

        # Orchestrator виконує:
        # - формування промпту;
        # - виклик AI-провайдера;
        # - перевірку JSON;
        # - перевірку джерел;
        # - повтор при невалідній відповіді.

        result = self.orchestrator.answer(
            data.question,
            data.documents,
            context,
            request_id,
        )

        # ======================================
        # 5. Завершення сценарію
        # ======================================

        self.telemetry.emit(
            request_id,
            "service",
            "finished",
        )

        return result

    # ==========================================
    # ПР14–ПР16 — Фонове виконання запитів
    # ==========================================

    def run_job(
        self,
        data: AskInput,
        request_id: str,
    ):
        """
        Виконує AI-сценарій у фоновій задачі.

        Результат зберігається в SQLite.

        У разі збою зберігається контрольована
        помилка, яку потім повертає API.
        """

        # Позначаємо запит як активний.
        self.storage.set_running(request_id)

        try:
            # Викликаємо основний сценарій.
            result = self.handle_question(
                data,
                request_id,
            )

            # Зберігаємо перевірену відповідь.
            self.storage.complete(
                request_id,
                result.model_dump(),
            )

        except AppError as err:

            # Контрольована помилка.
            self.storage.fail(
                request_id,
                {
                    "kind": err.kind,
                    "message": err.message,
                    "retryable": err.retryable,
                    "status_code": err.status_code,
                },
            )

        except Exception:

            # Не передаємо користувачеві
            # внутрішній текст винятку.
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
                        "Внутрішня помилка сервера"
                    ),
                    "retryable": False,
                    "status_code": 503,
                },
            )
