
"""
ПР18 — Автоматизовані тести повторних спроб.

Перевіряємо:
- тимчасовий збій і успішний повтор;
- заборону повтору постійних помилок;
- Retry-After;
- обмеження максимальної кількості спроб;
- денний бюджет;
- загальний дедлайн;
- HTTP 429 від провайдера;
- перевірку конфігурації Gateway.
"""

import json

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

from app.providers.base import BaseProvider
from app.providers.openai_compat import (
    OpenAICompatibleProvider,
    parse_retry_after,
)

from app.settings import Settings
from app.telemetry import Telemetry


class SequenceProvider(BaseProvider):
    """
    Провайдер, який повертає результати
    в заданій послідовності.

    Можна передати:
    - AppError;
    - ProviderResult.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def stream(self, request):
        yield ""

    def complete(self, request):
        self.calls.append(request)

        if not self.outcomes:
            raise AssertionError(
                "Gateway зробив зайвий виклик."
            )

        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class CountingBudget:
    """Тестовий лічильник AI-викликів."""

    def __init__(self):
        self.calls = 0

    def charge(self):
        self.calls += 1


def make_request():
    return ProviderRequest(
        request_id="pr18-retry-test",
        model="mock-study-v1",
        documents=["data/notes.md"],
        timeout_s=10.0,
        messages=[
            ProviderMessage(
                role="user",
                content="Що робить git push?",
            )
        ],
    )


def successful_result():
    return ProviderResult(
        text=json.dumps(
            {
                "status": "ok",
                "answer": "git push надсилає коміти.",
                "sources": ["data/notes.md"],
            },
            ensure_ascii=False,
        ),
        model="mock-study-v1",
    )


def make_gateway(
    tmp_path,
    provider,
    *,
    budget=None,
    max_attempts=2,
    waits=None,
):
    if waits is None:
        waits = []

    telemetry = Telemetry(
        str(tmp_path / "retries.jsonl")
    )

    return Gateway(
        provider,
        telemetry,
        budget,
        max_attempts=max_attempts,
        base_delay_s=0.4,
        max_delay_s=2.0,
        sleep_fn=waits.append,
        random_fn=lambda: 1.0,
    )


def test_timeout_retried_and_succeeded(tmp_path):
    """
    Timeout, потім успіх.

    Має бути рівно два звернення.
    """

    provider = SequenceProvider(
        [
            AppError(
                "timeout",
                "Тимчасовий тайм-аут",
                504,
                True,
            ),
            successful_result(),
        ]
    )

    waits = []

    gateway = make_gateway(
        tmp_path,
        provider,
        waits=waits,
    )

    result = gateway.complete(
        make_request()
    )

    assert "git push" in result.text
    assert len(provider.calls) == 2

    assert waits == pytest.approx(
        [0.4]
    )

    log = (
        tmp_path / "retries.jsonl"
    ).read_text(encoding="utf-8")

    assert log.count("provider_call") == 2
    assert "retry_scheduled" in log
    assert "retry_succeeded" in log


def test_permanent_error_not_retried(tmp_path):
    """
    Помилка авторизації не повторюється,
    навіть якщо вона помилково позначена
    як retryable.
    """

    provider = SequenceProvider(
        [
            AppError(
                "auth",
                "Помилка авторизації",
                503,
                True,
            ),
        ]
    )

    waits = []

    gateway = make_gateway(
        tmp_path,
        provider,
        waits=waits,
    )

    with pytest.raises(AppError) as exc:
        gateway.complete(
            make_request()
        )

    assert exc.value.kind == "auth"
    assert len(provider.calls) == 1
    assert waits == []


def test_retryable_false_prevents_retry(tmp_path):
    """
    Ознака retryable=False
    забороняє повтор навіть для timeout.
    """

    provider = SequenceProvider(
        [
            AppError(
                "timeout",
                "Час вичерпано",
                504,
                False,
            ),
        ]
    )

    gateway = make_gateway(
        tmp_path,
        provider,
    )

    with pytest.raises(AppError):
        gateway.complete(
            make_request()
        )

    assert len(provider.calls) == 1


def test_retry_after_is_respected(tmp_path):
    """
    Пауза Retry-After має пріоритет
    над звичайним backoff.
    """

    error = AppError(
        "rate_limit",
        "Забагато запитів",
        429,
        True,
    )

    error.retry_after_s = 2.5

    provider = SequenceProvider(
        [
            error,
            successful_result(),
        ]
    )

    waits = []

    gateway = make_gateway(
        tmp_path,
        provider,
        waits=waits,
    )

    result = gateway.complete(
        make_request()
    )

    assert result.model == "mock-study-v1"
    assert len(provider.calls) == 2

    assert waits == pytest.approx(
        [2.5]
    )


def test_maximum_two_attempts(tmp_path):
    """
    Тривалий збій не повинен
    запускати нескінченні повтори.
    """

    errors = [
        AppError(
            "provider_unavailable",
            "Провайдер недоступний",
            503,
            True,
        )
        for _ in range(3)
    ]

    provider = SequenceProvider(
        errors
    )

    waits = []

    gateway = make_gateway(
        tmp_path,
        provider,
        waits=waits,
    )

    with pytest.raises(AppError) as exc:
        gateway.complete(
            make_request()
        )

    assert exc.value.kind == "provider_unavailable"
    assert len(provider.calls) == 2

    # Між двома спробами лише одна пауза.
    assert len(waits) == 1


def test_budget_charged_per_attempt(tmp_path):
    """
    Кожне реальне звернення до моделі
    повинно враховуватись у бюджеті.
    """

    provider = SequenceProvider(
        [
            AppError(
                "timeout",
                "Тимчасовий збій",
                504,
                True,
            ),
            successful_result(),
        ]
    )

    budget = CountingBudget()

    gateway = make_gateway(
        tmp_path,
        provider,
        budget=budget,
    )

    gateway.complete(
        make_request()
    )

    assert len(provider.calls) == 2
    assert budget.calls == 2


def test_deadline_prevents_long_retry(tmp_path):
    """
    Якщо Retry-After довший за залишок
    дедлайну, повтор не виконується.
    """

    error = AppError(
        "rate_limit",
        "Зачекайте перед повтором",
        429,
        True,
    )

    error.retry_after_s = 20.0

    provider = SequenceProvider(
        [
            error,
            successful_result(),
        ]
    )

    waits = []

    gateway = make_gateway(
        tmp_path,
        provider,
        waits=waits,
    )

    with request_deadline(1.0):
        with pytest.raises(AppError) as exc:
            gateway.complete(
                make_request()
            )

    assert exc.value.kind == "rate_limit"

    # Немає другої спроби.
    assert len(provider.calls) == 1
    assert waits == []


def test_provider_parses_retry_after():
    """
    Перевіряє обробку заголовка
    Retry-After у секундах.
    """

    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("0") == 0.0

    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("invalid") is None
    assert parse_retry_after("-1") is None


def test_real_http_429_exposes_retry_after():
    """
    OpenAI-сумісний провайдер отримує
    HTTP 429 із заголовком Retry-After.

    Зовнішня мережа не використовується.
    """

    def handler(request):
        return httpx.Response(
            status_code=429,
            headers={
                "Retry-After": "4",
            },
            json={
                "error": {
                    "message": "Rate limit",
                }
            },
        )

    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen3:4b",
    )

    provider = OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(
            handler
        ),
    )

    request = ProviderRequest(
        request_id="pr18-http-429",
        model="qwen3:4b",
        timeout_s=10,
        documents=["data/notes.md"],
        messages=[
            ProviderMessage(
                role="user",
                content="Тест",
            )
        ],
    )

    try:
        with pytest.raises(AppError) as exc:
            provider.complete(
                request
            )
    finally:
        provider.close()

    assert exc.value.kind == "rate_limit"
    assert exc.value.status_code == 429
    assert exc.value.retryable is True

    assert getattr(
        exc.value,
        "retry_after_s",
        None,
    ) == 4.0


def test_invalid_retry_configuration(tmp_path):
    """
    Gateway повинен відхиляти
    некоректні налаштування.
    """

    provider = SequenceProvider(
        [successful_result()]
    )

    telemetry = Telemetry(
        str(tmp_path / "invalid.jsonl")
    )

    with pytest.raises(ValueError):
        Gateway(
            provider,
            telemetry,
            max_attempts=0,
        )

    with pytest.raises(ValueError):
        Gateway(
            provider,
            telemetry,
            max_attempts=6,
        )

    with pytest.raises(ValueError):
        Gateway(
            provider,
            telemetry,
            base_delay_s=-1,
        )

    with pytest.raises(ValueError):
        Gateway(
            provider,
            telemetry,
            base_delay_s=4,
            max_delay_s=2,
        )
