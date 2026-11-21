
"""
ПР18 — Тести безпечних помилок та AI_ENABLED.

Перевірки:
1. Правильне читання AI_ENABLED.
2. Некоректне значення AI_ENABLED.
3. Блокування AI-сценарію.
4. Збереження помилки ai_disabled.
5. Робота сервісу з увімкненим AI.
6. Збереження Retry-After.
7. Відхилення некоректного Retry-After.
8. Приховування неочікуваних винятків.
9. Обробка невідомого виду AppError.

Справжню Ollama не запускаємо.
"""

from types import SimpleNamespace

import pytest

from app.models import (
    AppError,
    AskInput,
    ModelOutput,
)
from app.service import StudyService
from app.settings import read_bool_env


class FakeStorage:
    """Тестове сховище без SQLite."""

    def __init__(self):
        self.calls = []
        self.status = None
        self.result = None
        self.error = None

    def set_running(self, request_id):
        self.calls.append("set_running")
        self.status = "running"

    def load_documents(
        self,
        documents,
        request_id,
    ):
        self.calls.append("load_documents")

        return (
            "<<<DOC:data/notes.md>>>\n"
            "git push надсилає коміти.\n"
            "<<<END_DOC>>>"
        )

    def complete(
        self,
        request_id,
        result,
    ):
        self.calls.append("complete")
        self.status = "completed"
        self.result = result

    def fail(
        self,
        request_id,
        error,
    ):
        self.calls.append("fail")
        self.status = "failed"
        self.error = error


class FakeTelemetry:
    """Зберігає назви подій у пам'яті."""

    def __init__(self):
        self.events = []

    def emit(
        self,
        request_id,
        layer,
        event,
    ):
        self.events.append(
            (request_id, layer, event)
        )


class FakeOrchestrator:
    """Модель без зовнішніх HTTP-запитів."""

    def __init__(
        self,
        enabled=True,
        error=None,
    ):
        self.settings = SimpleNamespace(
            ai_enabled=enabled,
            rag_enabled=False,
            llm_timeout_s=10.0,
        )

        self.calls = 0
        self.error = error

    def answer(
        self,
        question,
        documents,
        context,
        request_id,
    ):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return ModelOutput(
            status="ok",
            answer=(
                "git push надсилає коміти."
            ),
            sources=["data/notes.md"],
        )


def make_service(
    enabled=True,
    error=None,
):
    storage = FakeStorage()
    telemetry = FakeTelemetry()

    orchestrator = FakeOrchestrator(
        enabled=enabled,
        error=error,
    )

    service = StudyService(
        storage,
        orchestrator,
        telemetry,
    )

    return service, storage, orchestrator, telemetry


def make_question():
    return AskInput(
        question="Що робить git push?",
        documents=["data/notes.md"],
    )


def test_ai_enabled_parsing(monkeypatch):
    """Перевіряємо true та false."""

    monkeypatch.setenv(
        "AI_ENABLED",
        "false",
    )

    assert read_bool_env(
        "AI_ENABLED",
        True,
    ) is False

    monkeypatch.setenv(
        "AI_ENABLED",
        "true",
    )

    assert read_bool_env(
        "AI_ENABLED",
        False,
    ) is True


def test_invalid_ai_enabled_value(monkeypatch):
    """Некоректне значення не ігнорується."""

    monkeypatch.setenv(
        "AI_ENABLED",
        "possibly",
    )

    with pytest.raises(ValueError):
        read_bool_env(
            "AI_ENABLED",
            True,
        )


def test_disabled_ai_blocks_entire_scenario():
    """
    AI_ENABLED=false повинен зупинити
    роботу до читання документів та RAG.
    """

    service, storage, orchestrator, telemetry = (
        make_service(enabled=False)
    )

    with pytest.raises(AppError) as exc:
        service.handle_question(
            make_question(),
            request_id="pr18-disabled",
        )

    assert exc.value.kind == "ai_disabled"
    assert exc.value.status_code == 503
    assert exc.value.retryable is False

    assert storage.calls == []
    assert orchestrator.calls == 0

    assert (
        "pr18-disabled",
        "service",
        "ai_disabled",
    ) in telemetry.events


def test_disabled_ai_error_saved():
    """
    Фонове завдання з вимкненим AI
    зберігає контрольовану помилку.
    """

    service, storage, orchestrator, _ = (
        make_service(enabled=False)
    )

    service.run_job(
        make_question(),
        "pr18-background-disabled",
    )

    assert storage.status == "failed"

    assert storage.error["kind"] == "ai_disabled"
    assert storage.error["status_code"] == 503
    assert storage.error["retryable"] is False

    assert orchestrator.calls == 0

    assert "load_documents" not in storage.calls


def test_enabled_ai_completes_successfully():
    """
    Вимикач не порушує звичайний
    сценарій ПР15.
    """

    service, storage, orchestrator, telemetry = (
        make_service(enabled=True)
    )

    service.run_job(
        make_question(),
        "pr18-enabled",
    )

    assert storage.status == "completed"
    assert storage.error is None

    assert storage.result["status"] == "ok"

    assert (
        storage.result["sources"]
        == ["data/notes.md"]
    )

    assert orchestrator.calls == 1

    assert (
        "pr18-enabled",
        "service",
        "finished",
    ) in telemetry.events


def test_retry_after_saved_without_secret():
    """
    Контрольована помилка зберігає
    Retry-After, але не сторонній
    текст провайдера.
    """

    secret = "SECRET-DO-NOT-LOG-123"

    error = AppError(
        "rate_limit",
        (
            "Провайдер повернув: "
            + secret
        ),
        429,
        True,
    )

    error.retry_after_s = 4.2

    service, storage, _, _ = make_service(
        error=error
    )

    service.run_job(
        make_question(),
        "pr18-retry-after",
    )

    assert storage.status == "failed"

    assert storage.error["kind"] == "rate_limit"

    assert storage.error["status_code"] == 429
    assert storage.error["retryable"] is True

    assert storage.error["retry_after_s"] == 4.2

    assert secret not in str(
        storage.error
    )


@pytest.mark.parametrize(
    "retry_after",
    [
        float("nan"),
        float("inf"),
        -4,
        "invalid",
        True,
    ],
)
def test_invalid_retry_after_not_saved(
    retry_after,
):
    """
    У SQLite не потрапляє некоректний
    або нескінченний час очікування.
    """

    error = AppError(
        "rate_limit",
        "Тимчасове обмеження",
        429,
        True,
    )

    error.retry_after_s = retry_after

    payload = StudyService._error_payload(
        error
    )

    assert "retry_after_s" not in payload


def test_unexpected_error_is_safe():
    """
    Внутрішні подробиці Python
    не повинні з'являтися в SQLite.
    """

    secret = "SECRET-IN-EXCEPTION-456"

    service, storage, _, telemetry = make_service(
        error=RuntimeError(
            "Database crashed: " + secret
        )
    )

    service.run_job(
        make_question(),
        "pr18-internal",
    )

    assert storage.status == "failed"

    assert storage.error["kind"] == "internal_error"
    assert storage.error["status_code"] == 500
    assert storage.error["retryable"] is False

    assert secret not in str(storage.error)

    assert (
        "pr18-internal",
        "service",
        "internal_error",
    ) in telemetry.events


def test_unknown_app_error_becomes_internal():
    """
    Невідомий вид AppError не повинен
    розкривати внутрішній текст.
    """

    error = AppError(
        "unknown_secret_error",
        "SECRET-PRIVATE-DATA",
        503,
        True,
    )

    payload = StudyService._error_payload(
        error
    )

    assert payload["kind"] == "internal_error"
    assert payload["status_code"] == 500
    assert payload["retryable"] is False

    assert "SECRET" not in str(payload)
