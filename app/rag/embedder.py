
from __future__ import annotations

from math import isfinite
from typing import Sequence

import httpx


class EmbeddingError(RuntimeError):
    """Контрольована помилка компонента embeddings."""


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "qwen3-embedding:0.6b",
    base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    """
    Створює вектори текстів через локальну Ollama.

    Один текст -> один вектор.
    Порожній список -> порожній результат.
    """

    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts має бути списком текстів")

    if not texts:
        return []

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Кожен текст має бути непорожнім рядком"
        )

    if not isinstance(model, str) or not model.strip():
        raise ValueError("Потрібно вказати модель")

    if timeout_s <= 0:
        raise ValueError("Таймаут має бути додатним")

    url = f"{base_url.rstrip('/')}/api/embed"

    payload = {
        "model": model,
        "input": list(texts),
    }

    def request_vectors(
        http_client: httpx.Client,
    ) -> list[list[float]]:
        try:
            response = http_client.post(
                url,
                json=payload,
                timeout=timeout_s,
            )

            response.raise_for_status()

            result = response.json()

        except (
            httpx.TimeoutException,
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValueError,
        ) as exc:
            raise EmbeddingError(
                "Не вдалося отримати embeddings з Ollama."
            ) from exc

        if not isinstance(result, dict):
            raise EmbeddingError(
                "Некоректний формат відповіді Ollama."
            )

        raw_vectors = result.get("embeddings")

        if (
            not isinstance(raw_vectors, list)
            or len(raw_vectors) != len(texts)
        ):
            raise EmbeddingError(
                "Кількість векторів не відповідає "
                "кількості текстів."
            )

        vectors: list[list[float]] = []

        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingError(
                    "Ollama повернула порожній або "
                    "некоректний вектор."
                )

            vector: list[float] = []

            for value in raw_vector:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                ):
                    raise EmbeddingError(
                        "Вектор містить некоректне число."
                    )

                vector.append(float(value))

            vectors.append(vector)

        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Вектори мають різну розмірність."
            )

        return vectors

    if client is not None:
        return request_vectors(client)

    with httpx.Client(timeout=timeout_s) as http_client:
        return request_vectors(http_client)
