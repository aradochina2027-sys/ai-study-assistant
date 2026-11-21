
"""
ПР15–ПР18 — OpenAI-сумісний AI-провайдер.

Можливості:
- Ollama та OpenAI-сумісні HTTP API;
- контрольований тайм-аут;
- загальний дедлайн AI-сценарію;
- визначення тимчасових помилок;
- підтримка Retry-After;
- структурована відповідь;
- вимірювання часу й токенів.

Автоматичні повтори виконуються в Gateway,
а не всередині HTTP-клієнта.
"""

from collections.abc import Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
from time import perf_counter

import httpx

from app.deadlines import (
    check_deadline,
    http_timeout,
)
from app.models import (
    AppError,
    ProviderRequest,
    ProviderResult,
)
from app.providers.base import BaseProvider
from app.settings import Settings


def parse_retry_after(
    value: str | None,
) -> float | None:
    """
    HTTP Retry-After може містити:
    - кількість секунд;
    - дату у форматі HTTP-date.

    Повертає кількість секунд очікування.
    """

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    # --------------------------------------
    # Варіант 1. Кількість секунд
    # --------------------------------------

    try:
        seconds = float(value)

        if (
            not isfinite(seconds)
            or seconds < 0
        ):
            return None

        return seconds

    except ValueError:
        pass

    # --------------------------------------
    # Варіант 2. HTTP-date
    # --------------------------------------

    try:
        retry_time = parsedate_to_datetime(
            value
        )

        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(
            timezone.utc
        )

        seconds = (
            retry_time - now
        ).total_seconds()

        return max(
            0.0,
            seconds,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def provider_error(
    kind: str,
    message: str,
    status_code: int,
    retryable: bool,
    retry_after: float | None = None,
) -> AppError:
    """
    Створює контрольовану помилку.

    retry_after_s передається до Gateway,
    але не розкриває конфіденційних даних.
    """

    error = AppError(
        kind=kind,
        message=message,
        status_code=status_code,
        retryable=retryable,
    )

    if retry_after is not None:
        error.retry_after_s = retry_after

    return error


class OpenAICompatibleProvider(BaseProvider):

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings

        self.client = httpx.Client(
            base_url=(
                settings.llm_base_url.rstrip("/")
                + "/"
            ),
            timeout=http_timeout(
                settings.llm_timeout_s
            ),
            transport=transport,
            headers={
                "Authorization": (
                    "Bearer "
                    + (
                        settings.llm_api_key
                        or "ollama"
                    )
                )
            },
        )

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Один HTTP-виклик AI-провайдера.

        Повторів усередині немає.

        Gateway вирішує, чи потрібно
        повторювати помилку.
        """

        started = perf_counter()

        check_deadline()

        configured_timeout = min(
            self.settings.llm_timeout_s,
            request.timeout_s,
        )

        timeout = http_timeout(
            configured_timeout
        )

        messages = [
            message.model_dump()
            for message in request.messages
        ]

        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": {
                "type": "json_object"
            },
            "stream": False,
        }

        # Для локальної Ollama вимикаємо
        # окремий reasoning-режим.
        if self.settings.llm_provider in {
            "ollama",
            "recorded",
        }:
            payload["reasoning_effort"] = "none"

        try:
            response = self.client.post(
                "chat/completions",
                json=payload,
                timeout=timeout,
            )

            response.raise_for_status()

            data = response.json()

        # ======================================
        # 1. Тайм-аут
        # ======================================

        except httpx.TimeoutException as exc:
            check_deadline()

            raise provider_error(
                "timeout",
                "Модель не відповіла вчасно.",
                504,
                True,
            ) from exc

        # ======================================
        # 2. Збій мережі
        # ======================================

        except httpx.RequestError as exc:
            check_deadline()

            raise provider_error(
                "provider_unavailable",
                "Не вдалося підключитися "
                "до AI-провайдера.",
                503,
                True,
            ) from exc

        # ======================================
        # 3. HTTP-помилки провайдера
        # ======================================

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code

            retry_after = parse_retry_after(
                exc.response.headers.get(
                    "Retry-After"
                )
            )

            if status == 429:
                raise provider_error(
                    "rate_limit",
                    "AI-провайдер тимчасово "
                    "обмежив кількість запитів.",
                    429,
                    True,
                    retry_after,
                ) from exc

            if status in (401, 403):
                raise provider_error(
                    "auth",
                    "Не вдалося авторизуватися "
                    "в AI-провайдера.",
                    503,
                    False,
                ) from exc

            if status in (400, 413, 422):
                raise provider_error(
                    "bad_request",
                    "AI-провайдер відхилив запит.",
                    503,
                    False,
                ) from exc

            if status >= 500:
                raise provider_error(
                    "provider_unavailable",
                    "AI-провайдер тимчасово "
                    "недоступний.",
                    503,
                    True,
                    retry_after,
                ) from exc

            raise provider_error(
                "provider_error",
                "Не вдалося виконати "
                "запит до AI-провайдера.",
                503,
                False,
            ) from exc

        # ======================================
        # 4. Некоректний JSON відповіді
        # ======================================

        except ValueError as exc:
            raise provider_error(
                "provider_invalid_response",
                "AI-провайдер повернув "
                "некоректну JSON-відповідь.",
                503,
                False,
            ) from exc

        check_deadline()

        # ======================================
        # 5. Перевірка структури відповіді
        # ======================================

        try:
            if not isinstance(data, dict):
                raise ValueError(
                    "JSON-відповідь не є об'єктом."
                )

            choices = data["choices"]

            message = choices[0]["message"]

            content = (
                message["content"] or ""
            )

            if not isinstance(content, str):
                raise ValueError(
                    "Некоректний тип вмісту."
                )

            usage = data.get("usage") or {}

            result = ProviderResult(
                text=content,
                model=(
                    data.get("model")
                    or request.model
                ),
                input_tokens=int(
                    usage.get(
                        "prompt_tokens"
                    ) or 0
                ),
                output_tokens=int(
                    usage.get(
                        "completion_tokens"
                    ) or 0
                ),
                elapsed_ms=round(
                    (
                        perf_counter() - started
                    ) * 1000,
                    2,
                ),
            )

        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise provider_error(
                "provider_invalid_response",
                "Не вдалося прочитати "
                "відповідь AI.",
                503,
                False,
            ) from exc

        return result

    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:
        """
        Сумісність із BaseProvider.

        Поточний StudyAI повертає
        перевірений результат цілком.
        """

        result = self.complete(
            request
        )

        yield result.text

    def close(self) -> None:
        """Закриває HTTP-клієнт."""

        self.client.close()
