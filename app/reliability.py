
"""
ПР18 — Механізми надійності AI Study Assistant.

CircuitBreaker:
    Захищає провайдера від повторних звернень
    під час тривалої недоступності.

SlidingWindowRateLimiter:
    Обмежує кількість запитів окремого клієнта
    за визначений проміжок часу.

Обидва компоненти потокобезпечні.
"""

from collections import deque
from math import ceil
from threading import Lock
from time import monotonic
from typing import Callable

from app.models import AppError


AVAILABILITY_ERRORS = frozenset({
    "timeout",
    "provider_unavailable",
    "rate_limit",
})


class CircuitBreaker:
    """
    Стани:
        closed — звичайна робота;
        open — виклики заблоковано;
        half_open — дозволено один пробний виклик.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 20.0,
        clock: Callable[[], float] = monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError(
                "failure_threshold має бути >= 1"
            )

        if recovery_timeout_s <= 0:
            raise ValueError(
                "recovery_timeout_s має бути > 0"
            )

        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._clock = clock

        self._lock = Lock()

        self._state = "closed"
        self._failures = 0
        self._opened_until = 0.0
        self._probe_active = False

    def _open(self) -> None:
        """Викликати тільки під блокуванням."""

        self._state = "open"
        self._opened_until = (
            self._clock() + self.recovery_timeout_s
        )
        self._probe_active = False

    def _close(self) -> None:
        """Викликати тільки під блокуванням."""

        self._state = "closed"
        self._failures = 0
        self._opened_until = 0.0
        self._probe_active = False

    def state(self) -> str:
        """Повертає актуальний стан запобіжника."""

        with self._lock:
            if (
                self._state == "open"
                and self._clock() >= self._opened_until
            ):
                return "half_open"

            return self._state

    def snapshot(self) -> dict:
        """
        Показники для /api/health.

        Метод не викликає AI-провайдера.
        """

        with self._lock:
            remaining = 0.0

            if self._state == "open":
                remaining = max(
                    0.0,
                    self._opened_until - self._clock(),
                )

            current = self._state

            if current == "open" and remaining == 0:
                current = "half_open"

            return {
                "state": current,
                "failures": self._failures,
                "failure_threshold": (
                    self.failure_threshold
                ),
                "retry_after_s": round(
                    remaining,
                    2,
                ),
            }

    def before_call(self) -> bool:
        """
        Перевіряє дозвіл на звернення.

        Повертає True для пробного виклику
        в стані half_open.

        Повертає False у звичайному стані.

        Якщо запобіжник відкритий,
        генерує контрольовану AppError.
        """

        with self._lock:
            now = self._clock()

            if self._state == "open":
                remaining = (
                    self._opened_until - now
                )

                if remaining > 0:
                    error = AppError(
                        "circuit_open",
                        "AI-провайдер тимчасово "
                        "недоступний. Спробуйте пізніше.",
                        503,
                        True,
                    )

                    error.retry_after_s = remaining
                    raise error

                self._state = "half_open"
                self._probe_active = False

            if self._state == "half_open":
                if self._probe_active:
                    error = AppError(
                        "circuit_open",
                        "Провайдер уже проходить "
                        "перевірку доступності.",
                        503,
                        True,
                    )
                    error.retry_after_s = 1.0
                    raise error

                self._probe_active = True
                return True

            return False

    def record_success(
        self,
        probe: bool = False,
    ) -> None:
        """Успішний виклик скидає лічильник."""

        with self._lock:
            self._close()

    def record_failure(
        self,
        error: AppError,
        probe: bool = False,
    ) -> None:
        """
        Рахує тільки збої доступності.

        Некоректний запит і невалідна відповідь
        не повинні розмикати запобіжник.
        """

        with self._lock:
            if error.kind not in AVAILABILITY_ERRORS:
                if probe:
                    self._close()
                else:
                    self._failures = 0

                return

            if probe:
                self._open()
                return

            if self._state != "closed":
                return

            self._failures += 1

            if self._failures >= self.failure_threshold:
                self._open()

    def release_probe(self) -> None:
        """
        Звільняє пробний виклик, якщо
        він не дійшов до провайдера.
        """

        with self._lock:
            if self._state == "half_open":
                self._probe_active = False


class SlidingWindowRateLimiter:
    """
    Ліміт за ковзним часовим вікном.

    Приклад:
        10 запитів за 60 секунд
        для окремого client_id.

    Облік виконується у пам'яті процесу.
    """

    def __init__(
        self,
        max_requests: int = 10,
        window_s: float = 60.0,
        clock: Callable[[], float] = monotonic,
    ):
        if max_requests < 1:
            raise ValueError(
                "max_requests має бути >= 1"
            )

        if window_s <= 0:
            raise ValueError(
                "window_s має бути > 0"
            )

        self.max_requests = max_requests
        self.window_s = window_s
        self._clock = clock

        self._lock = Lock()
        self._requests: dict[
            str,
            deque[float],
        ] = {}

    def check(self, client_id: str) -> None:
        """
        Реєструє один дозволений запит.

        Коли ліміт вичерпано:
            HTTP 429;
            retryable=True;
            retry_after_s.
        """

        if not client_id:
            raise ValueError(
                "client_id не може бути порожнім"
            )

        now = self._clock()

        with self._lock:
            timestamps = self._requests.setdefault(
                client_id,
                deque(),
            )

            cutoff = now - self.window_s

            while (
                timestamps
                and timestamps[0] <= cutoff
            ):
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                oldest = timestamps[0]

                retry_after = max(
                    0.0,
                    oldest + self.window_s - now,
                )

                error = AppError(
                    "client_rate_limit",
                    "Забагато запитів. "
                    "Спробуйте трохи пізніше.",
                    429,
                    True,
                )

                error.retry_after_s = max(
                    1,
                    ceil(retry_after),
                )

                raise error

            timestamps.append(now)

    def remaining(self, client_id: str) -> int:
        """Кількість доступних запитів."""

        now = self._clock()

        with self._lock:
            timestamps = self._requests.get(
                client_id,
            )

            if timestamps is None:
                return self.max_requests

            cutoff = now - self.window_s

            while (
                timestamps
                and timestamps[0] <= cutoff
            ):
                timestamps.popleft()

            return max(
                0,
                self.max_requests - len(timestamps),
            )
