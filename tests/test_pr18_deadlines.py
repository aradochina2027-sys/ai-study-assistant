
"""
ПР18 — Тести тайм-аутів і загального дедлайну.

Перевіряємо:
1. Створення дедлайну.
2. Залишок часу.
3. Завершення дедлайну.
4. Вкладені дедлайни.
5. HTTP-тайм-аути LLM.
6. HTTP-тайм-аути embeddings.
7. Контроль часу в StudyService.
8. Перевірку після повільного провайдера.
"""

from types import SimpleNamespace
from time import sleep

import httpx
import pytest

import app.deadlines as deadlines

from app.models import (
    AppError,
    AskInput,
    ProviderMessage,
    ProviderRequest,
)

from app.providers.openai_compat import (
    OpenAICompatibleProvider,
)

from app.rag.embedder import embed_texts
from app.service import StudyService
from app.settings import Settings


def test_deadline_remaining(monkeypatch):
    """Залишок часу зменшується."""

    now = [100.0]

    monkeypatch.setattr(
        deadlines,
        "monotonic",
        lambda: now[0],
    )

    with deadlines.request_deadline(10):
        assert deadlines.remaining_seconds() == 10

        now[0] = 103.0

        assert deadlines.remaining_seconds() == 7

        assert deadlines.bounded_timeout(20) == 7

        assert deadlines.bounded_timeout(2) == 2


def test_deadline_exceeded(monkeypatch):
    """Після дедлайну повертається 504."""

    now = [50.0]

    monkeypatch.setattr(
        deadlines,
        "monotonic",
        lambda: now[0],
    )

    with deadlines.request_deadline(5):
        now[0] = 56.0

        with pytest.raises(AppError) as exc:
            deadlines.check_deadline()

        assert exc.value.kind == "deadline_exceeded"
        assert exc.value.status_code == 504
        assert exc.value.retryable is False


def test_nested_deadline_cannot_extend(monkeypatch):
    """Вкладений сценарій не продовжує час."""

    now = [100.0]

    monkeypatch.setattr(
        deadlines,
        "monotonic",
        lambda: now[0],
    )

    with deadlines.request_deadline(10):
        now[0] = 103.0

        with deadlines.request_deadline(100):
            assert deadlines.remaining_seconds() == 7

        assert deadlines.remaining_seconds() == 7

    assert deadlines.remaining_seconds() is None


def test_invalid_timeout():
    """Нульовий або від'ємний час заборонений."""

    for value in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            deadlines.bounded_timeout(value)


def test_http_timeout_limited_by_deadline():
    """HTTPX отримує обмежений тайм-аут."""

    with deadlines.request_deadline(0.5):
        timeout = deadlines.http_timeout(120)

        assert 0 < timeout.connect <= 0.5
        assert 0 < timeout.read <= 0.5
        assert 0 < timeout.write <= 0.5
        assert 0 < timeout.pool <= 0.5


def test_llm_uses_remaining_deadline():
    """
    LLM отримує залишок загального часу.

    Справжня Ollama не викликається.
    """

    captured = {}

    def handler(request):
        captured["timeout"] = request.extensions.get(
            "timeout"
        )

        return httpx.Response(
            200,
            json={
                "model": "qwen3:4b",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"status":"ok",'
                                '"answer":"Тест",'
                                '"sources":[]}'
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            },
        )

    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen3:4b",
    )

    provider = OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(handler),
    )

    request = ProviderRequest(
        request_id="pr18-llm",
        model="qwen3:4b",
        documents=["data/notes.md"],
        messages=[
            ProviderMessage(
                role="user",
                content="Тест",
            )
        ],
        timeout_s=120,
    )

    try:
        with deadlines.request_deadline(1.0):
            result = provider.complete(request)
    finally:
        provider.client.close()

    assert result.model == "qwen3:4b"

    timeout = captured["timeout"]

    assert timeout is not None
    assert 0 < timeout["read"] <= 1.0


def test_embeddings_use_remaining_deadline():
    """
    Embeddings отримують залишок часу
    без справжнього виклику Ollama.
    """

    captured = {}

    def handler(request):
        captured["timeout"] = request.extensions.get(
            "timeout"
        )

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2, 0.3]
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        transport=transport
    ) as client:

        with deadlines.request_deadline(1.0):
            result = embed_texts(
                ["Навчальний текст"],
                client=client,
                timeout_s=120,
            )

    assert result == [[0.1, 0.2, 0.3]]

    timeout = captured["timeout"]

    assert timeout is not None
    assert 0 < timeout["read"] <= 1.0


def test_service_stops_before_model():
    """
    Якщо час вичерпано на підготовці
    документів, LLM не запускається.
    """

    calls = []

    class FakeStorage:
        def load_documents(
            self,
            documents,
            request_id,
        ):
            sleep(0.03)

            return (
                "<<<DOC:data/notes.md>>>\n"
                "Тест\n"
                "<<<END_DOC>>>"
            )

    class FakeOrchestrator:
        settings = SimpleNamespace(
            rag_enabled=False,
            llm_timeout_s=0.01,
        )

        def answer(self, *args):
            calls.append("llm")
            raise AssertionError(
                "LLM не повинна запускатися."
            )

    class FakeTelemetry:
        def emit(self, *args):
            pass

    service = StudyService(
        FakeStorage(),
        FakeOrchestrator(),
        FakeTelemetry(),
    )

    data = AskInput(
        question="Що таке Python?",
    )

    with pytest.raises(AppError) as exc:
        service.handle_question(
            data,
            request_id="pr18-slow",
        )

    assert exc.value.kind == "deadline_exceeded"
    assert calls == []


def test_deadline_checked_after_slow_llm():
    """
    Навіть якщо тестовий транспорт ігнорує
    тайм-аут, результат після дедлайну
    не вважається успішним.

    Це перевіряє контроль після виклику,
    а не фізичне переривання транспорту.
    """

    def handler(request):
        sleep(0.03)

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "{}"
                        }
                    }
                ]
            },
        )

    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen3:4b",
    )

    provider = OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(handler),
    )

    request = ProviderRequest(
        request_id="pr18-expired",
        model="qwen3:4b",
        documents=["data/notes.md"],
        messages=[
            ProviderMessage(
                role="user",
                content="Тест",
            )
        ],
        timeout_s=120,
    )

    try:
        with pytest.raises(AppError) as exc:
            with deadlines.request_deadline(0.01):
                provider.complete(request)
    finally:
        provider.client.close()

    assert exc.value.kind == "deadline_exceeded"
