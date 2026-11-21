
"""ПР18: контрольовані навмисні збої без запуску Ollama."""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.deadlines import request_deadline
from app.gateway import Gateway
from app.models import (
    AppError,
    ProviderMessage,
    ProviderRequest,
    ProviderResult,
)
from app.rag.cache import EmbeddingCache
from app.rag.embedder import (
    EmbeddingError,
    embed_texts,
)
from app.reliability import CircuitBreaker
from app.telemetry import Telemetry


class SequenceProvider:
    """Підставний AI-провайдер."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1

        result = self.outcomes.pop(0)

        if isinstance(result, Exception):
            raise result

        return result


class BudgetCounter:
    """Підставний лічильник бюджету."""

    def __init__(self):
        self.charges = 0

    def charge(self):
        self.charges += 1


def request():
    return ProviderRequest(
        request_id=str(uuid.uuid4()),
        model="mock-study-v1",
        timeout_s=10,
        documents=["data/notes.md"],
        prompt_version="v2",
        messages=[
            ProviderMessage(
                role="user",
                content="PRIVATE QUESTION: git push",
            )
        ],
    )


def good():
    return ProviderResult(
        text=(
            '{"status":"ok",'
            '"answer":"test",'
            '"sources":[]}'
        ),
        model="mock-study-v1",
        input_tokens=8,
        output_tokens=4,
    )


def error(
    kind="timeout",
    retryable=True,
):
    return AppError(
        kind,
        "SECRET-ERROR-TEXT",
        504 if kind == "timeout" else 503,
        retryable,
    )


def gateway(
    tmp_path,
    provider,
    *,
    attempts=2,
    breaker=None,
    budget=None,
    sleep=None,
):
    return Gateway(
        provider,
        Telemetry(
            str(tmp_path / "faults.jsonl")
        ),
        budget,
        max_attempts=attempts,
        breaker=breaker,
        sleep_fn=(
            sleep
            if sleep is not None
            else lambda _: None
        ),
        random_fn=lambda: 1.0,
        base_delay_s=0.2,
    )


def test_timeout_then_recovery_and_no_secret_in_log(
    tmp_path,
):
    """Тайм-аут, повтор, успіх і безпечний журнал."""

    fake = SequenceProvider([
        error(),
        good(),
    ])

    budget = BudgetCounter()

    app_gateway = gateway(
        tmp_path,
        fake,
        budget=budget,
    )

    result = app_gateway.complete(
        request()
    )

    assert result.model == "mock-study-v1"
    assert fake.calls == 2
    assert budget.charges == 2

    raw = (
        tmp_path / "faults.jsonl"
    ).read_text(encoding="utf-8")

    assert "SECRET-ERROR-TEXT" not in raw
    assert "PRIVATE QUESTION" not in raw

    assert raw.count(
        '"event":"provider_call"'
    ) == 2

    assert '"event":"retry_succeeded"' in raw


def test_auth_is_not_retried(tmp_path):
    """Помилка авторизації не повторюється."""

    fake = SequenceProvider([
        error(
            "auth",
            retryable=False,
        ),
        good(),
    ])

    app_gateway = gateway(
        tmp_path,
        fake,
    )

    with pytest.raises(AppError) as exc:
        app_gateway.complete(
            request()
        )

    assert exc.value.kind == "auth"
    assert fake.calls == 1


def test_retry_after_honored(tmp_path):
    """Gateway дотримується Retry-After."""

    temporary = AppError(
        "rate_limit",
        "Rate limit",
        429,
        True,
    )

    temporary.retry_after_s = 3.5

    fake = SequenceProvider([
        temporary,
        good(),
    ])

    waits = []

    app_gateway = gateway(
        tmp_path,
        fake,
        sleep=waits.append,
    )

    app_gateway.complete(
        request()
    )

    assert fake.calls == 2

    assert waits == pytest.approx([
        3.5
    ])


def test_deadline_prevents_retry_after(
    tmp_path,
):
    """Повтор не запускається після дедлайну."""

    temporary = AppError(
        "rate_limit",
        "Rate limit",
        429,
        True,
    )

    temporary.retry_after_s = 100.0

    fake = SequenceProvider([
        temporary,
        good(),
    ])

    app_gateway = gateway(
        tmp_path,
        fake,
    )

    with request_deadline(0.5):
        with pytest.raises(AppError) as exc:
            app_gateway.complete(
                request()
            )

    assert exc.value.kind == "rate_limit"
    assert fake.calls == 1


def test_breaker_opens_blocks_then_recovers(
    tmp_path,
):
    """Circuit Breaker відкривається та відновлюється."""

    clock = [100.0]

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_s=5,
        clock=lambda: clock[0],
    )

    fake = SequenceProvider([
        error(),
        error(),
        good(),
    ])

    app_gateway = gateway(
        tmp_path,
        fake,
        attempts=1,
        breaker=breaker,
    )

    for _ in range(2):
        with pytest.raises(AppError):
            app_gateway.complete(
                request()
            )

    assert breaker.state() == "open"

    with pytest.raises(AppError) as exc:
        app_gateway.complete(
            request()
        )

    assert exc.value.kind == "circuit_open"
    assert fake.calls == 2

    clock[0] = 106.0

    app_gateway.complete(
        request()
    )

    assert fake.calls == 3
    assert breaker.state() == "closed"


def test_rag_embedding_unavailable_does_not_write_cache(
    monkeypatch,
):
    """Помилка embeddings не потрапляє до кешу."""

    monkeypatch.setenv(
        "STUDYAI_EMBED_CACHE_ENABLED",
        "1",
    )

    cache = EmbeddingCache()

    def http_error(_request):
        return httpx.Response(503)

    transport = httpx.MockTransport(
        http_error
    )

    with httpx.Client(
        transport=transport
    ) as client:

        with pytest.raises(EmbeddingError):
            embed_texts(
                ["doc text"],
                client=client,
                cache=cache,
                cache_owner="studyai-local",
            )

    assert cache.stats()["entries"] == 0


def test_cache_hit_and_bypass_without_ollama(
    monkeypatch,
):
    """Повторне звернення використовує кеш."""

    monkeypatch.setenv(
        "STUDYAI_EMBED_CACHE_ENABLED",
        "1",
    )

    calls = []

    def respond(http_request):
        calls.append(1)

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [1.0, 2.0]
                ]
            },
        )

    cache = EmbeddingCache()

    transport = httpx.MockTransport(
        respond
    )

    with httpx.Client(
        transport=transport
    ) as client:

        kwargs = {
            "client": client,
            "cache": cache,
            "cache_owner": "studyai-local",
        }

        first = embed_texts(
            ["same"],
            **kwargs,
        )

        second = embed_texts(
            ["same"],
            **kwargs,
        )

        third = embed_texts(
            ["same"],
            bypass_cache=True,
            **kwargs,
        )

    assert first == second == third
    assert len(calls) == 2
    assert cache.stats()["hits"] == 1
