
"""ПР18 — Безпечний JSONL-журнал із ротацією.

Записуються лише дозволені службові поля.
Не передавати сюди промпти, відповіді, ключі,
тексти документів або повідомлення винятків.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid

from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any


_IDENTIFIER = re.compile(
    r"^[a-z][a-z0-9_]{0,63}$"
)

_MODEL = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,79}$"
)

_VERSION = re.compile(
    r"^v[0-9]+(?:[._-][A-Za-z0-9]+)?$"
)

_MODEL_PREFIXES = (
    "qwen",
    "mock-",
    "gpt-",
    "gemini-",
    "claude-",
    "llama",
    "deepseek",
    "mistral",
    "phi",
    "o1",
    "o3",
    "o4",
    "nomic-",
    "text-embedding",
    "embeddinggemma",
)

_PROVIDERS = frozenset({
    "ollama",
    "openai_compat",
    "recorded",
    "mock",
    "unknown",
})

_PROVIDER_KINDS = frozenset({
    "main",
    "fallback",
    "cache",
    "degraded",
    "mock",
    "recorded",
})

_STATUSES = frozenset({
    "started",
    "ok",
    "error",
    "retry",
    "rejected",
    "skipped",
    "completed",
    "failed",
    "cancelled",
    "open",
    "closed",
})

_ERROR_KINDS = frozenset({
    "timeout",
    "provider_unavailable",
    "rate_limit",
    "auth",
    "bad_request",
    "provider_error",
    "provider_invalid_response",
    "invalid_output",
    "deadline_exceeded",
    "rag_unavailable",
    "budget",
    "circuit_open",
    "concurrency_limit",
    "client_rate_limit",
    "ai_disabled",
    "internal_error",
    "validation_error",
    "idempotency_conflict",
    "invalid_idempotency_key",
    "not_found",
    "conflict",
    "document_missing",
    "invalid_document",
    "document_too_large",
    "empty_documents",
    "rag_document_unsupported",
})

_INT_FIELDS = frozenset({
    "attempt",
    "input_tokens",
    "output_tokens",
})

_FLOAT_FIELDS = frozenset({
    "duration_ms",
    "ttft_ms",
    "cost",
    "backoff_ms",
})

_BOOL_FIELDS = frozenset({
    "cached",
    "fallback",
})


def _safe_request_id(
    value: object,
) -> str:
    """
    Зберігає UUID для наскрізного пошуку.

    Довільні значення хешуються, щоб
    випадково не записати особисті дані.
    """

    raw = str(value)

    try:
        parsed = uuid.UUID(raw)

        if str(parsed) == raw.lower():
            return raw

    except (
        ValueError,
        AttributeError,
        TypeError,
    ):
        pass

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]

    return "hash_" + digest


def _identifier(
    value: object,
    fallback: str,
) -> str:
    """
    Дозволяє лише короткі технічні назви.

    Наприклад:
        provider_call
        provider_returned
        retry_scheduled

    Текст документа не допускається.
    """

    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
    ):
        return fallback

    forbidden_words = (
        "secret",
        "password",
        "apikey",
        "token",
        "question",
    )

    if any(
        word in value
        for word in forbidden_words
    ):
        return fallback

    return value


def _safe_metadata(
    name: str,
    value: Any,
) -> Any:
    """
    Перевіряє значення дозволеного поля.

    Невідомі, нескінченні та некоректні
    значення відхиляються.
    """

    if name in _INT_FIELDS:
        if (
            type(value) is int
            and 0 <= value <= 1_000_000_000
        ):
            return value

    elif name in _FLOAT_FIELDS:
        if (
            type(value) in (int, float)
            and math.isfinite(value)
            and 0 <= value <= 1_000_000_000
        ):
            return round(
                float(value),
                4,
            )

    elif name in _BOOL_FIELDS:
        if type(value) is bool:
            return value

    elif name == "provider":
        if (
            isinstance(value, str)
            and value in _PROVIDERS
        ):
            return value

    elif name == "provider_kind":
        if (
            isinstance(value, str)
            and value in _PROVIDER_KINDS
        ):
            return value

    elif name == "status":
        if (
            isinstance(value, str)
            and value in _STATUSES
        ):
            return value

    elif name == "error_kind":
        if (
            isinstance(value, str)
            and value in _ERROR_KINDS
        ):
            return value

    elif name == "model":
        if (
            isinstance(value, str)
            and _MODEL.fullmatch(value)
            and value.lower().startswith(
                _MODEL_PREFIXES
            )
        ):
            return value

        return "other_model"

    elif name in (
        "prompt_version",
        "schema_version",
    ):
        if (
            isinstance(value, str)
            and _VERSION.fullmatch(value)
        ):
            return value

    return None


class _JSONFormatter(
    logging.Formatter
):
    """
    Один об'єкт JSON на один рядок.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        return json.dumps(
            record.msg,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


class Telemetry:
    """
    Структурований JSONL-журнал.

    Старий інтерфейс збережено:

        emit(request_id, layer, event)

    Новий інтерфейс:

        emit(
            request_id,
            layer,
            event,
            attempt=1,
            duration_ms=100,
        )
    """

    def __init__(
        self,
        filename: str,
        *,
        max_bytes: int = 2_000_000,
        backup_count: int = 3,
    ):
        if (
            max_bytes < 128
            or backup_count < 1
        ):
            raise ValueError(
                "max_bytes має бути >= 128, "
                "backup_count має бути >= 1."
            )

        self.filename = Path(
            filename
        )

        self.filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.lock = Lock()

        self._logger = logging.Logger(
            (
                "studyai.telemetry."
                + uuid.uuid4().hex
            ),
            level=logging.INFO,
        )

        self._logger.propagate = False

        # ПР18 — автоматична ротація.
        self._handler = RotatingFileHandler(
            self.filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )

        self._handler.setFormatter(
            _JSONFormatter()
        )

        self._logger.addHandler(
            self._handler
        )

    def emit(
        self,
        request_id: str,
        layer: str,
        event: str,
        **fields: Any,
    ) -> None:
        """
        Записує тільки дозволені метадані.

        Поля на кшталт:
            question
            password
            api_key
            messages
            prompt
            documents

        не потрапляють у журнал.
        """

        entry: dict[str, Any] = {
            "ts": datetime.now(
                timezone.utc
            ).isoformat(),

            "request_id": _safe_request_id(
                request_id
            ),

            "layer": _identifier(
                layer,
                "other",
            ),

            "event": _identifier(
                event,
                "event_rejected",
            ),
        }

        allowed_fields = (
            "provider",
            "model",
            "prompt_version",
            "schema_version",
            "attempt",
            "duration_ms",
            "ttft_ms",
            "status",
            "error_kind",
            "cached",
            "fallback",
            "provider_kind",
            "input_tokens",
            "output_tokens",
            "cost",
            "backoff_ms",
        )

        for name in allowed_fields:
            if name not in fields:
                continue

            value = _safe_metadata(
                name,
                fields[name],
            )

            if value is not None:
                entry[name] = value

        with self.lock:
            self._logger.info(entry)

    def close(self) -> None:
        """
        Закриває обробник журналу.
        """

        with self.lock:
            self._logger.removeHandler(
                self._handler
            )

            self._handler.close()
