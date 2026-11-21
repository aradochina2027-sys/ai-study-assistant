
from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path

import httpx

from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts


INDEX_VERSION = 1
DEFAULT_MODEL = "qwen3-embedding:0.6b"


def load_index(index_path: str | Path) -> dict:
    """Завантажує та перевіряє векторний індекс."""

    path = Path(index_path)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Некоректний формат індексу")

    if data.get("schema_version") != INDEX_VERSION:
        raise ValueError("Непідтримувана версія індексу")

    chunks = data.get("chunks")
    dimensions = data.get("dimensions")

    if not isinstance(chunks, list):
        raise ValueError("В індексі немає списку фрагментів")

    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 0
    ):
        raise ValueError("Некоректна розмірність векторів")

    if chunks and dimensions == 0:
        raise ValueError("Індекс містить фрагменти без векторів")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Некоректний фрагмент")

        for field in ("chunk_id", "source", "text"):
            if not isinstance(chunk.get(field), str):
                raise ValueError(
                    f"Некоректне поле фрагмента: {field}"
                )

        vector = chunk.get("embedding")

        if not isinstance(vector, list):
            raise ValueError("Відсутній вектор фрагмента")

        if len(vector) != dimensions:
            raise ValueError("Неправильна розмірність вектора")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in vector
        ):
            raise ValueError("Вектор містить некоректні числа")

    return data


def build_index(
    document_path: str | Path = "data/notes.md",
    index_path: str | Path = ".rag_index/notes.json",
    *,
    model: str = DEFAULT_MODEL,
    max_chars: int = 900,
    overlap_chars: int = 120,
    client: httpx.Client | None = None,
) -> dict:
    """
    Створює або повторно використовує індекс документа.

    Якщо документ, модель і параметри не змінилися,
    повторна векторизація не виконується.
    """

    document = Path(document_path)
    index_file = Path(index_path)

    if not document.is_file():
        raise FileNotFoundError(
            f"Документ не знайдено: {document}"
        )

    if not model.strip():
        raise ValueError("Назва моделі не може бути порожньою")

    raw_bytes = document.read_bytes()

    document_text = raw_bytes.decode("utf-8-sig")
    document_hash = sha256(raw_bytes).hexdigest()

    document_name = document.as_posix()

    # Перевіряємо, чи можна повторно використати індекс.
    if index_file.is_file():
        try:
            existing = load_index(index_file)

            same_document = (
                existing.get("document") == document_name
                and existing.get("document_sha256")
                == document_hash
            )

            same_settings = (
                existing.get("embedding_model") == model
                and existing.get("max_chars") == max_chars
                and existing.get("overlap_chars")
                == overlap_chars
            )

            if same_document and same_settings:
                return existing

        except (ValueError, OSError, json.JSONDecodeError):
            # Пошкоджений або застарілий індекс
            # створюємо заново.
            pass

    # Розбиваємо документ на фрагменти.
    chunks = chunk_text(
        document_text,
        source=document_name,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    # Отримуємо вектори всіх фрагментів.
    if chunks:
        vectors = embed_texts(
            [chunk.text for chunk in chunks],
            model=model,
            client=client,
        )
    else:
        vectors = []

    dimensions = len(vectors[0]) if vectors else 0

    # Формуємо записи векторного індексу.
    records = []

    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "embedding": vector,
            }
        )

    index_data = {
        "schema_version": INDEX_VERSION,
        "document": document_name,
        "document_sha256": document_hash,
        "embedding_model": model,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "dimensions": dimensions,
        "chunks": records,
    }

    # Створюємо папку, якщо вона відсутня.
    index_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Спочатку записуємо тимчасовий файл,
    # потім замінюємо ним основний.
    temporary_file = index_file.with_name(
        index_file.name + ".tmp"
    )

    temporary_file.write_text(
        json.dumps(
            index_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(index_file)

    return index_data
