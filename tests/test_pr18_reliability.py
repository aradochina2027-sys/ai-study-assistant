
"""
ПР18 — Тести Circuit Breaker,
обмеження одночасності та частоти запитів.

Справжню Ollama не викликаємо.
"""

from threading import Event, Thread

import pytest

from app.gateway import Gateway
from app.models import (
    AppError,
    ProviderMessage,
    ProviderRequest,
    ProviderResult,
)
from app.reliability import (
    CircuitBreaker,
    SlidingWindowRateLimiter,
)
from app.telemetry import Telemetry


def temporary_error():
    return AppError(
        "provider_unavailable",
        "Провайдер недоступний",
        503,
        True,
    )


def successful_result():
    return ProviderResult(
        text='{"status":"ok","answer":"Тест","sources":[]}',
        model="mock-study-v1",
    )


def make_request():
    return ProviderRequest(
        request_id="pr18-reliability",
        model="mock-study-v1",
        timeout_s=10,
        documents=["data/notes.md"],
        messages=[
            ProviderMessage(
                role="user",
                content="Тестове запитання",
            )
        ],
    )


class FakeProvider:
    def __init__(self, failures=0):
        self.calls = 0
        self.failures = failures

    def complete(self, request):
        self.calls += 1

        if self.calls <= self.failures:
            raise temporary_error()

        return successful_result()


class FakeBudget:
    def __init__(self):
        self.charges = 0

    def charge(self):
        self.charges += 1


def test_breaker_opens_after_threshold():
    """
    Два збої доступності відкривають
    Circuit Breaker.
    """

    now = [100.0]

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_s=10,
        clock=lambda: now[0],
    )

    for _ in range(2):
        probe = breaker.before_call()

        assert probe is False

        breaker.record_failure(
            temporary_error(),
            probe,
        )

    assert breaker.state() == "open"

    with pytest.raises(AppError) as exc:
        breaker.before_call()

    assert exc.value.kind == "circuit_open"
    assert exc.value.status_code == 503
    assert exc.value.retryable is True

    assert exc.value.retry_after_s == 10


def test_breaker_half_open_and_recovery():
    """
    Після завершення паузи дозволено
    один пробний виклик.
    """

    now = [100.0]

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_s=5,
        clock=lambda: now[0],
    )

    breaker.record_failure(
        temporary_error()
    )

    assert breaker.state() == "open"

    now[0] = 106.0

    assert breaker.state() == "half_open"

    probe = breaker.before_call()

    assert probe is True

    breaker.record_success(probe)

    assert breaker.state() == "closed"

    assert (
        breaker.snapshot()["failures"]
        == 0
    )


def test_breaker_only_one_half_open_probe():
    """Пробних викликів одночасно може бути один."""

    now = [10.0]

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_s=2,
        clock=lambda: now[0],
    )

    breaker.record_failure(
        temporary_error()
    )

    now[0] = 13.0

    first_probe = breaker.before_call()

    assert first_probe is True

    with pytest.raises(AppError) as exc:
        breaker.before_call()

    assert exc.value.kind == "circuit_open"

    breaker.record_success(first_probe)

    assert breaker.state() == "closed"


def test_non_availability_error_does_not_open():
    """
    Некоректні дані не вважаємо
    збоєм доступності провайдера.
    """

    breaker = CircuitBreaker(
        failure_threshold=1,
    )

    breaker.record_failure(
        AppError(
            "bad_request",
            "Некоректний запит",
            400,
            False,
        )
    )

    assert breaker.state() == "closed"

    assert breaker.snapshot()["failures"] == 0


def test_rate_limit_blocks_extra_request():
    """
    За одне часовe вікно дозволено
    лише два запити одного клієнта.
    """

    now = [100.0]

    limiter = SlidingWindowRateLimiter(
        max_requests=2,
        window_s=60,
        clock=lambda: now[0],
    )

    limiter.check("client-A")
    limiter.check("client-A")

    assert limiter.remaining("client-A") == 0

    with pytest.raises(AppError) as exc:
        limiter.check("client-A")

    assert exc.value.kind == "client_rate_limit"
    assert exc.value.status_code == 429
    assert exc.value.retry_after_s == 60

    # Інший клієнт має окремий ліміт.
    limiter.check("client-B")

    assert limiter.remaining("client-B") == 1


def test_rate_limit_recovers_after_window():
    """Після завершення вікна ліміт оновлюється."""

    now = [100.0]

    limiter = SlidingWindowRateLimiter(
        max_requests=1,
        window_s=10,
        clock=lambda: now[0],
    )

    limiter.check("client-A")

    assert limiter.remaining("client-A") == 0

    now[0] = 111.0

    assert limiter.remaining("client-A") == 1

    limiter.check("client-A")

    assert limiter.remaining("client-A") == 0


def test_gateway_uses_circuit_breaker(tmp_path):
    """
    Після відкриття запобіжника
    нові виклики провайдера блокуються.

    Заблокований виклик не списує бюджет.
    """

    provider = FakeProvider(
        failures=10
    )

    budget = FakeBudget()

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_s=60,
    )

    telemetry = Telemetry(
        str(tmp_path / "breaker.jsonl")
    )

    gateway = Gateway(
        provider,
        telemetry,
        budget,
        breaker=breaker,
        max_attempts=1,
        sleep_fn=lambda _: None,
    )

    request = make_request()

    for _ in range(2):
        with pytest.raises(AppError) as exc:
            gateway.complete(request)

        assert exc.value.kind == "provider_unavailable"

    assert breaker.state() == "open"

    with pytest.raises(AppError) as exc:
        gateway.complete(request)

    assert exc.value.kind == "circuit_open"

    assert provider.calls == 2
    assert budget.charges == 2


def test_gateway_limits_concurrency(tmp_path):
    """
    Другий запит отримує контрольовану
    помилку, якщо перший зайняв єдине місце.
    """

    entered = Event()
    release = Event()

    class BlockingProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            entered.set()

            if not release.wait(timeout=3):
                raise AssertionError(
                    "Тестовий виклик завис."
                )

            return successful_result()

    provider = BlockingProvider()

    telemetry = Telemetry(
        str(tmp_path / "concurrency.jsonl")
    )

    gateway = Gateway(
        provider,
        telemetry,
        max_attempts=1,
        max_concurrent=1,
        queue_timeout_s=0,
    )

    first_result = []
    first_error = []

    def first_request():
        try:
            first_result.append(
                gateway.complete(
                    make_request()
                )
            )
        except Exception as exc:
            first_error.append(exc)

    thread = Thread(
        target=first_request,
    )

    thread.start()

    try:
        assert entered.wait(timeout=3)

        with pytest.raises(AppError) as exc:
            gateway.complete(
                make_request()
            )

        assert exc.value.kind == "concurrency_limit"
        assert exc.value.status_code == 429

    finally:
        release.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert first_error == []
    assert len(first_result) == 1
    assert provider.calls == 1


def test_gateway_health_without_provider_call(tmp_path):
    """
    Перевірка стану не повинна
    викликати AI-модель.
    """

    provider = FakeProvider()

    telemetry = Telemetry(
        str(tmp_path / "health.jsonl")
    )

    gateway = Gateway(
        provider,
        telemetry,
        max_concurrent=2,
    )

    health = gateway.health()

    assert health["breaker"]["state"] == "closed"
    assert health["max_concurrent"] == 2
    assert provider.calls == 0


def test_breaker_reopens_if_probe_fails():
    """
    Якщо пробний виклик знову завершується
    збоєм, запобіжник відкривається повторно.
    """

    now = [100.0]

    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_s=5,
        clock=lambda: now[0],
    )

    breaker.record_failure(
        temporary_error()
    )

    now[0] = 106.0

    probe = breaker.before_call()

    breaker.record_failure(
        temporary_error(),
        probe,
    )

    assert breaker.state() == "open"

    assert (
        breaker.snapshot()["retry_after_s"]
        == 5.0
    )
