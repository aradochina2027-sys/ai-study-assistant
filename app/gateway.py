
"""ПР18 — Gateway із повним журналюванням.

Можливості:
- контрольовані повтори;
- exponential backoff та jitter;
- Retry-After;
- спільний дедлайн;
- денний бюджет;
- Circuit Breaker;
- обмеження одночасності;
- безпечна телеметрія викликів моделі.
"""

from __future__ import annotations

import random
import time

from math import isfinite
from threading import BoundedSemaphore
from typing import Callable

from app.budget import DailyBudget
from app.deadlines import (
    bounded_timeout,
    check_deadline,
    remaining_seconds,
)
from app.models import (
    AppError,
    ProviderRequest,
    ProviderResult,
)
from app.providers.base import BaseProvider
from app.reliability import CircuitBreaker
from app.telemetry import Telemetry


RETRYABLE_KINDS = frozenset({
    "timeout",
    "rate_limit",
    "provider_unavailable",
})


class Gateway:
    def __init__(
        self,
        provider: BaseProvider,
        telemetry: Telemetry,
        budget: DailyBudget | None = None,
        *,
        max_attempts: int = 2,
        base_delay_s: float = 0.25,
        max_delay_s: float = 3.0,
        sleep_fn: Callable[[float], None] | None = None,
        random_fn: Callable[[], float] | None = None,
        breaker: CircuitBreaker | None = None,
        max_concurrent: int = 2,
        queue_timeout_s: float = 0.5,
    ):
        if (
            type(max_attempts) is not int
            or not 1 <= max_attempts <= 5
        ):
            raise ValueError(
                "max_attempts має бути від 1 до 5"
            )

        if (
            not isfinite(base_delay_s)
            or base_delay_s < 0
        ):
            raise ValueError(
                "Некоректний base_delay_s"
            )

        if (
            not isfinite(max_delay_s)
            or max_delay_s < base_delay_s
        ):
            raise ValueError(
                "Некоректний max_delay_s"
            )

        if (
            type(max_concurrent) is not int
            or max_concurrent < 1
        ):
            raise ValueError(
                "max_concurrent має бути >= 1"
            )

        if (
            not isfinite(queue_timeout_s)
            or queue_timeout_s < 0
        ):
            raise ValueError(
                "Некоректний queue_timeout_s"
            )

        self.provider = provider
        self.telemetry = telemetry
        self.budget = budget

        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s

        self._sleep = (
            sleep_fn or time.sleep
        )

        self._random = (
            random_fn or random.random
        )

        self.breaker = (
            breaker
            if breaker is not None
            else CircuitBreaker()
        )

        self.max_concurrent = max_concurrent
        self.queue_timeout_s = queue_timeout_s

        self._semaphore = BoundedSemaphore(
            max_concurrent
        )

    # ==========================================
    # 1. Стан Gateway
    # ==========================================

    def health(self) -> dict:
        """
        Не викликає Ollama.
        """

        return {
            "breaker": self.breaker.snapshot(),
            "max_concurrent": self.max_concurrent,
            "queue_timeout_s": self.queue_timeout_s,
        }

    # ==========================================
    # 2. Назва провайдера
    # ==========================================

    def _provider_name(self) -> str:
        """
        Для журналу використовуємо
        назву типу провайдера, а не
        URL, ключ або текст запиту.
        """

        settings = getattr(
            self.provider,
            "settings",
            None,
        )

        configured = getattr(
            settings,
            "llm_provider",
            None,
        )

        if configured in {
            "mock",
            "ollama",
            "openai_compat",
            "recorded",
        }:
            return configured

        class_name = (
            type(self.provider).__name__.lower()
        )

        for prefix, label in (
            ("mock", "mock"),
            ("recorded", "recorded"),
            ("openai", "openai_compat"),
        ):
            if class_name.startswith(prefix):
                return label

        return "unknown"

    # ==========================================
    # 3. Безпечний запис подій
    # ==========================================

    def _log(
        self,
        request: ProviderRequest,
        event: str,
        attempt: int,
        **fields,
    ) -> None:
        """
        У журнал потрапляють лише
        службові метадані.

        НЕ передаються:
        - request.messages;
        - request.documents;
        - result.text;
        - текст винятків;
        - API-ключі.
        """

        metadata = {
            "provider": self._provider_name(),
            "model": request.model,
            "prompt_version": (
                request.prompt_version
            ),
            "schema_version": "v1",
            "attempt": attempt,
        }

        metadata.update(fields)

        self.telemetry.emit(
            request.request_id,
            "gateway",
            event,
            **metadata,
        )

    # ==========================================
    # 4. Exponential backoff + jitter
    # ==========================================

    def _retry_delay(
        self,
        attempt: int,
        error: AppError,
    ) -> float:
        exponential = min(
            self.max_delay_s,
            self.base_delay_s * (
                2 ** (attempt - 1)
            ),
        )

        try:
            random_value = float(
                self._random()
            )
        except (TypeError, ValueError):
            random_value = 0.0

        if not isfinite(random_value):
            random_value = 0.0

        random_value = min(
            1.0,
            max(0.0, random_value),
        )

        delay = exponential * (
            0.5 + random_value * 0.5
        )

        retry_after = getattr(
            error,
            "retry_after_s",
            None,
        )

        if retry_after is not None:
            try:
                provider_delay = float(
                    retry_after
                )
            except (TypeError, ValueError):
                provider_delay = 0.0

            if (
                isfinite(provider_delay)
                and provider_delay > 0
            ):
                delay = max(
                    delay,
                    provider_delay,
                )

        return delay

    # ==========================================
    # 5. Чи можна повторити помилку
    # ==========================================

    def _can_retry(
        self,
        error: AppError,
        attempt: int,
    ) -> bool:
        return (
            attempt < self.max_attempts
            and error.retryable
            and error.kind in RETRYABLE_KINDS
        )

    # ==========================================
    # 6. Обмеження одночасності
    # ==========================================

    def _acquire_slot(
        self,
        request: ProviderRequest,
    ) -> None:
        check_deadline()

        remaining = remaining_seconds()

        timeout = self.queue_timeout_s

        if remaining is not None:
            timeout = min(
                timeout,
                remaining,
            )

        acquired = self._semaphore.acquire(
            timeout=max(0.0, timeout)
        )

        if acquired:
            return

        check_deadline()

        self._log(
            request,
            "concurrency_limit",
            0,
            status="rejected",
            error_kind="concurrency_limit",
        )

        error = AppError(
            "concurrency_limit",
            (
                "Усі місця для AI-запитів зайняті. "
                "Спробуйте пізніше."
            ),
            429,
            True,
        )

        error.retry_after_s = 1.0

        raise error

    # ==========================================
    # 7. Основний виклик провайдера
    # ==========================================

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Послідовність:
        1. Дедлайн.
        2. Семафор.
        3. Circuit Breaker.
        4. Денний бюджет.
        5. Запис початку виклику.
        6. AI-провайдер.
        7. Запис результату.
        8. Повтор, якщо дозволено.
        """

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            check_deadline()

            attempt_request = (
                request.model_copy(
                    update={
                        "timeout_s": bounded_timeout(
                            request.timeout_s
                        ),
                    }
                )
            )

            # Отримуємо місце для AI-виклику.
            self._acquire_slot(
                request
            )

            error: AppError | None = None

            try:
                # --------------------------------
                # Circuit Breaker
                # --------------------------------

                probe = self.breaker.before_call()

                # --------------------------------
                # Денний бюджет
                # --------------------------------

                try:
                    if self.budget is not None:
                        self.budget.charge()

                except Exception:
                    self.breaker.release_probe()

                    self._log(
                        request,
                        "budget_rejected",
                        attempt,
                        status="rejected",
                    )

                    raise

                # --------------------------------
                # Початок фактичного AI-виклику
                # --------------------------------

                self._log(
                    request,
                    "provider_call",
                    attempt,
                    status="started",
                    cached=False,
                    fallback=False,
                )

                previous_state = (
                    self.breaker.state()
                )

                started = time.perf_counter()

                try:
                    result = self.provider.complete(
                        attempt_request
                    )

                # --------------------------------
                # Контрольований збій
                # --------------------------------

                except AppError as exc:
                    error = exc

                    elapsed_ms = (
                        time.perf_counter() - started
                    ) * 1000

                    self.breaker.record_failure(
                        exc,
                        probe,
                    )

                    self._log(
                        request,
                        f"error_{exc.kind}",
                        attempt,
                        status="error",
                        error_kind=exc.kind,
                        duration_ms=elapsed_ms,
                    )

                    current_state = (
                        self.breaker.state()
                    )

                    if (
                        previous_state != current_state
                        and current_state == "open"
                    ):
                        self._log(
                            request,
                            "breaker_open",
                            attempt,
                            status="open",
                        )

                # --------------------------------
                # Неочікувана помилка Python
                # --------------------------------

                except Exception:
                    self.breaker.release_probe()

                    self._log(
                        request,
                        "provider_exception",
                        attempt,
                        status="error",
                        error_kind="internal_error",
                    )

                    # Текст винятку не записуємо.
                    raise

                # --------------------------------
                # Успішна відповідь
                # --------------------------------

                else:
                    elapsed_ms = (
                        time.perf_counter() - started
                    ) * 1000

                    self.breaker.record_success(
                        probe
                    )

                    check_deadline()

                    self._log(
                        request,
                        "provider_returned",
                        attempt,
                        status="ok",
                        duration_ms=elapsed_ms,
                        model=result.model,
                        input_tokens=(
                            result.input_tokens
                        ),
                        output_tokens=(
                            result.output_tokens
                        ),
                        cached=False,
                        fallback=False,
                    )

                    if attempt > 1:
                        self._log(
                            request,
                            "retry_succeeded",
                            attempt,
                            status="ok",
                        )

                    if previous_state == "half_open":
                        self._log(
                            request,
                            "breaker_closed",
                            attempt,
                            status="closed",
                        )

                    return result

            finally:
                # Місце звільняється після
                # успіху або будь-якої помилки.
                self._semaphore.release()

            # ------------------------------------
            # Перевірка повтору
            # ------------------------------------

            if error is None:
                raise RuntimeError(
                    "Gateway завершив виклик "
                    "без результату."
                )

            if not self._can_retry(
                error,
                attempt,
            ):
                raise error

            if self.breaker.state() == "open":
                raise error

            delay_s = self._retry_delay(
                attempt,
                error,
            )

            remaining = remaining_seconds()

            if (
                remaining is not None
                and delay_s >= remaining
            ):
                self._log(
                    request,
                    "retry_skipped_deadline",
                    attempt,
                    status="skipped",
                )

                raise error

            # ------------------------------------
            # Подія запланованого повтору
            # ------------------------------------

            self._log(
                request,
                "retry_scheduled",
                attempt,
                status="retry",
                backoff_ms=delay_s * 1000,
            )

            if delay_s > 0:
                self._sleep(delay_s)

        raise RuntimeError(
            "Gateway завершився без "
            "результату або помилки."
        )
