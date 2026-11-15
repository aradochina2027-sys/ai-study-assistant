from collections.abc import Iterator
from time import perf_counter

import httpx

from app.models import (
    AppError,
    ProviderRequest,
    ProviderResult,
)
from app.providers.base import BaseProvider
from app.settings import Settings


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings

        self.client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/") + "/",
            timeout=settings.llm_timeout_s,
            transport=transport,
            headers={
                "Authorization": (
                    f"Bearer {settings.llm_api_key or 'ollama'}"
                )
            },
        )

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:
        start = perf_counter()

        payload = {
            "model": request.model,
            "messages": [
                message.model_dump()
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": {
                "type": "json_object"
            },
            "stream": False,
        }

        if self.settings.llm_provider in {
            "ollama", "recorded"
        }:
            payload["reasoning_effort"] = "none"

        try:
            response = self.client.post(
                "chat/completions",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as error:
            raise AppError(
                "timeout",
                "Модель не відповіла вчасно",
                504,
                True,
            ) from error

        except httpx.RequestError as error:
            raise AppError(
                "provider_unavailable",
                "Сервер моделі недоступний",
                503,
                True,
            ) from error

        except httpx.HTTPStatusError as error:
            code = error.response.status_code

            if code == 429:
                raise AppError(
                    "rate_limit",
                    "Ліміт запитів AI",
                    429,
                    True,
                ) from error

            if code in (401, 403):
                raise AppError(
                    "auth",
                    "Помилка авторизації AI",
                    503,
                ) from error

            if code == 400:
                raise AppError(
                    "bad_request",
                    "Модель не прийняла запит",
                    503,
                ) from error

            raise AppError(
                "provider_unavailable",
                "Помилка сервера моделі",
                503,
                True,
            ) from error

        except ValueError as error:
            raise AppError(
                "provider_invalid_response",
                "Некоректна відповідь сервера AI",
                503,
            ) from error

        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}

            return ProviderResult(
                text=text,
                model=data.get("model") or request.model,
                input_tokens=int(
                    usage.get("prompt_tokens") or 0
                ),
                output_tokens=int(
                    usage.get("completion_tokens") or 0
                ),
                elapsed_ms=round(
                    (perf_counter() - start) * 1000,
                    2,
                ),
            )

        except (
            ValueError, KeyError, IndexError, TypeError
        ) as error:
            raise AppError(
                "provider_invalid_response",
                "Не вдалося прочитати відповідь AI",
                503,
            ) from error

    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:
        yield self.complete(request).text