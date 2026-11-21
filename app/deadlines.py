
"""
ПР18 — Загальний дедлайн AI-сценарію.

Один дедлайн використовується для:
- завантаження навчального контексту;
- семантичного пошуку RAG;
- створення embeddings;
- викликів основної AI-моделі;
- перевірки результату.

ContextVar ізолює дедлайни різних запитів.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from math import isfinite
from time import monotonic
from typing import Iterator

import httpx

from app.models import AppError


_current_deadline: ContextVar[float | None] = ContextVar(
    "studyai_request_deadline",
    default=None,
)


def _validate_timeout(seconds: float) -> float:
    value = float(seconds)

    if not isfinite(value) or value <= 0:
        raise ValueError(
            "Тайм-аут повинен бути додатним "
            "скінченним числом."
        )

    return value


@contextmanager
def request_deadline(
    timeout_s: float,
) -> Iterator[float]:
    """
    Встановлює кінцевий момент виконання.

    Вкладений дедлайн не може продовжити
    дедлайн зовнішнього сценарію.
    """

    seconds = _validate_timeout(timeout_s)

    new_deadline = monotonic() + seconds
    previous = _current_deadline.get()

    if previous is not None:
        new_deadline = min(
            new_deadline,
            previous,
        )

    token = _current_deadline.set(
        new_deadline
    )

    try:
        yield new_deadline
    finally:
        _current_deadline.reset(token)


def remaining_seconds() -> float | None:
    """
    Повертає залишок загального часу.

    None означає, що дедлайн не встановлено.
    """

    deadline = _current_deadline.get()

    if deadline is None:
        return None

    return max(
        0.0,
        deadline - monotonic(),
    )


def check_deadline() -> None:
    """
    Перевіряє, чи не вичерпано дедлайн.
    """

    remaining = remaining_seconds()

    if remaining is not None and remaining <= 0:
        raise AppError(
            "deadline_exceeded",
            "Перевищено загальний час "
            "обробки запитання.",
            504,
            False,
        )


def bounded_timeout(
    configured_timeout_s: float,
) -> float:
    """
    Обмежує тайм-аут окремої операції
    залишком загального дедлайну.
    """

    configured = _validate_timeout(
        configured_timeout_s
    )

    check_deadline()

    remaining = remaining_seconds()

    if remaining is None:
        return configured

    return min(
        configured,
        remaining,
    )


def http_timeout(
    timeout_s: float,
) -> httpx.Timeout:
    """
    Створює HTTPX-тайм-аути.

    connect — встановлення з'єднання;
    read — очікування даних;
    write — надсилання даних;
    pool — отримання з'єднання з пулу.
    """

    seconds = bounded_timeout(timeout_s)

    return httpx.Timeout(
        connect=min(5.0, seconds),
        read=seconds,
        write=min(10.0, seconds),
        pool=min(5.0, seconds),
    )
