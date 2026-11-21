
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import httpx

from app.rag.embedder import embed_texts
from app.rag.index import load_index


@dataclass(frozen=True)
class SearchHit:
    """Один знайдений фрагмент документа."""

    chunk_id: str
    source: str
    text: str
    start: int
    end: int
    score: float


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Обчислює косинусну схожість двох векторів.

    Результат лежить у діапазоні від -1 до 1.
    Більше значення означає більшу схожість.
    """

    if not vector_a or not vector_b:
        raise ValueError("Вектори не можуть бути порожніми")

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Вектори повинні мати однакову розмірність"
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sqrt(sum(
        value * value
        for value in vector_a
    ))

    norm_b = sqrt(sum(
        value * value
        for value in vector_b
    ))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_index(
    question: str,
    *,
    index_path: str | Path = ".rag_index/notes.json",
    top_k: int = 3,
    min_score: float = 0.25,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> list[SearchHit]:
    """
    Шукає релевантні фрагменти у векторному індексі.

    1. Перетворює запитання на embedding.
    2. Порівнює його з векторами документів.
    3. Сортує фрагменти за схожістю.
    4. Повертає top_k найкращих результатів.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Запитання не може бути порожнім")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k має бути цілим числом")

    if top_k <= 0:
        raise ValueError("top_k має бути більше нуля")

    if not isinstance(min_score, (int, float)) or not isfinite(
        min_score
    ):
        raise ValueError("min_score має бути числом")

    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score має бути від -1 до 1")

    path = Path(index_path)

    if not path.is_file():
        raise FileNotFoundError(
            "Векторний індекс не знайдено. "
            "Спочатку виконай build_index()."
        )

    index_data = load_index(path)

    chunks = index_data["chunks"]

    if not chunks:
        return []

    model = index_data.get("embedding_model")

    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "В індексі не вказано модель embeddings"
        )

    # Для запитання використовуємо ту саму модель,
    # якою створювали вектори документів.
    query_vectors = embed_texts(
        [question.strip()],
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        client=client,
    )

    query_vector = query_vectors[0]

    if len(query_vector) != index_data["dimensions"]:
        raise ValueError(
            "Розмірність запитання не відповідає індексу. "
            "Перебудуй індекс відповідною моделлю."
        )

    results: list[SearchHit] = []

    for chunk in chunks:
        score = cosine_similarity(
            query_vector,
            chunk["embedding"],
        )

        if score < min_score:
            continue

        results.append(
            SearchHit(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"],
                text=chunk["text"],
                start=chunk["start"],
                end=chunk["end"],
                score=score,
            )
        )

    # Найбільш релевантні фрагменти — спочатку.
    results.sort(
        key=lambda hit: (
            -hit.score,
            hit.source,
            hit.start,
        )
    )

    return results[:top_k]
