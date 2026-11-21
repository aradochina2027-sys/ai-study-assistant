
"""ПР16–ПР18: embeddings через Ollama з опційним TTL-кешем.

Кеш вимкнений за замовчуванням. Для локального
однокористувацького проєкту встановіть:
STUDYAI_EMBED_CACHE_ENABLED=1
STUDYAI_EMBED_CACHE_OWNER=local-studyai
"""

from __future__ import annotations

import os

from math import isfinite
from typing import Sequence

import httpx

from app.deadlines import (
    check_deadline,
    http_timeout,
)
from app.rag.cache import EmbeddingCache


class EmbeddingError(RuntimeError):
    """Контрольована помилка embeddings."""


_DEFAULT_CACHE = EmbeddingCache(
    ttl_seconds=3600,
    max_entries=2048,
)


def embedding_cache_stats() -> dict[str, int | float]:
    return _DEFAULT_CACHE.stats()


def clear_embedding_cache() -> None:
    _DEFAULT_CACHE.clear()


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "qwen3-embedding:0.6b",
    base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
    cache_owner: str | None = None,
    bypass_cache: bool = False,
    cache: EmbeddingCache | None = None,
) -> list[list[float]]:
    """Повертає вектори в порядку вхідних текстів."""

    # ==========================================
    # 1. Перевірка вхідних даних
    # ==========================================

    if not isinstance(texts, (list, tuple)):
        raise TypeError(
            "texts має бути списком або кортежем"
        )

    if not texts:
        return []

    if any(
        not isinstance(text, str)
        or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Усі тексти мають бути непорожніми рядками"
        )

    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "Назва моделі не може бути порожньою"
        )

    if (
        not isinstance(timeout_s, (int, float))
        or not isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError(
            "timeout_s має бути додатним числом"
        )

    check_deadline()

    # ==========================================
    # 2. Налаштування кешу
    # ==========================================

    enabled = os.getenv(
        "STUDYAI_EMBED_CACHE_ENABLED",
        "0",
    ).lower() in {
        "1", "true", "yes", "on"
    }

    owner = (
        cache_owner
        if cache_owner is not None
        else os.getenv(
            "STUDYAI_EMBED_CACHE_OWNER",
            "",
        )
    )

    revision = os.getenv(
        "STUDYAI_EMBED_CACHE_MODEL_REVISION",
        "v1",
    )

    if enabled and not owner.strip():
        raise ValueError(
            "Задайте STUDYAI_EMBED_CACHE_OWNER "
            "для ізоляції кешу"
        )

    active_cache = (
        cache
        if cache is not None
        else _DEFAULT_CACHE
    )

    use_cache = enabled and not bypass_cache

    # ==========================================
    # 3. Пошук готових векторів у кеші
    # ==========================================

    vectors: list[list[float] | None] = [
        None
    ] * len(texts)

    missing: dict[str, list[int]] = {}

    for pos, text in enumerate(texts):
        if use_cache:
            saved = active_cache.get(
                text,
                model=model,
                owner_id=owner,
                revision=revision,
                base_url=base_url,
            )

            if saved is not None:
                vectors[pos] = saved
                continue

        # Якщо кеш увімкнений, однакові
        # тексти в межах однієї партії
        # відправляємо до моделі один раз.
        key = (
            text
            if use_cache
            else f"{pos}"
        )

        missing.setdefault(
            key,
            [],
        ).append(pos)

    # ==========================================
    # 4. Отримання відсутніх embeddings
    # ==========================================

    if missing:
        missing_texts = [
            texts[positions[0]]
            for positions in missing.values()
        ]

        payload = {
            "model": model,
            "input": list(missing_texts),
        }

        url = (
            base_url.rstrip("/")
            + "/api/embed"
        )

        def fetch(
            http_client: httpx.Client,
        ) -> list[list[float]]:
            check_deadline()

            try:
                response = http_client.post(
                    url,
                    json=payload,
                    timeout=http_timeout(timeout_s),
                )

                response.raise_for_status()

                result = response.json()

            except (
                httpx.TimeoutException,
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
            ) as exc:
                check_deadline()

                raise EmbeddingError(
                    "Не вдалося отримати embeddings "
                    "з Ollama"
                ) from exc

            check_deadline()

            if not isinstance(result, dict):
                raise EmbeddingError(
                    "Некоректна відповідь Ollama"
                )

            raw_vectors = result.get(
                "embeddings"
            )

            if (
                not isinstance(raw_vectors, list)
                or len(raw_vectors)
                != len(missing_texts)
            ):
                raise EmbeddingError(
                    "Кількість векторів не відповідає "
                    "кількості текстів"
                )

            received: list[list[float]] = []

            for raw in raw_vectors:
                if (
                    not isinstance(raw, list)
                    or not raw
                    or any(
                        isinstance(x, bool)
                        or not isinstance(
                            x,
                            (int, float),
                        )
                        or not isfinite(x)
                        for x in raw
                    )
                ):
                    raise EmbeddingError(
                        "Embedding-вектор містить "
                        "некоректні числа"
                    )

                received.append(
                    [
                        float(x)
                        for x in raw
                    ]
                )

            return received

        if client is None:
            with httpx.Client() as local_client:
                fetched = fetch(local_client)
        else:
            fetched = fetch(client)

        # Відновлюємо початковий порядок
        # усіх векторів.
        for positions, vector in zip(
            missing.values(),
            fetched,
        ):
            for pos in positions:
                vectors[pos] = vector.copy()

    # ==========================================
    # 5. Перевірка кінцевого результату
    # ==========================================

    checked = [
        vector
        for vector in vectors
        if vector is not None
    ]

    if (
        len(checked) != len(texts)
        or len({
            len(vector)
            for vector in checked
        }) != 1
    ):
        raise EmbeddingError(
            "Отримано embedding-вектори "
            "різної розмірності"
        )

    # ==========================================
    # 6. Збереження нових векторів у кеші
    # ==========================================

    if missing and use_cache:
        for text, vector in zip(
            missing_texts,
            fetched,
        ):
            active_cache.put(
                text,
                vector,
                model=model,
                owner_id=owner,
                revision=revision,
                base_url=base_url,
            )

    check_deadline()

    return checked
