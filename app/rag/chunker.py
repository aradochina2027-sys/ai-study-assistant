from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class TextChunk:
    """Окремий фрагмент навчального документа."""

    chunk_id: str
    source: str
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    """
    Розділяє документ на фрагменти.

    max_chars — максимальний розмір фрагмента.
    overlap_chars — перекриття сусідніх фрагментів.
    """

    if not isinstance(text, str):
        raise TypeError("text має бути рядком")

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Потрібно вказати джерело документа")

    if max_chars <= 0:
        raise ValueError("max_chars має бути більше 0")

    if not 0 <= overlap_chars < max_chars:
        raise ValueError(
            "overlap_chars має бути від 0 до max_chars - 1"
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    if not normalized.strip():
        return []

    chunks: list[TextChunk] = []

    start = 0
    length = len(normalized)

    while start < length:

        # Пропускаємо пробіли на початку фрагмента.
        while start < length and normalized[start].isspace():
            start += 1

        if start >= length:
            break

        end = min(start + max_chars, length)

        # За можливості завершуємо фрагмент
        # на межі абзацу, рядка або слова.
        if end < length:
            search_from = start + max_chars // 2

            for separator in ("\n\n", "\n", " "):
                position = normalized.rfind(
                    separator,
                    search_from,
                    end,
                )

                if position > start:
                    end = position
                    break

        raw = normalized[start:end]
        content = raw.strip()

        if content:
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())

            chunk_start = start + left_trim
            chunk_end = end - right_trim

            # Ідентифікатор залежить від документа,
            # позиції та вмісту фрагмента.
            raw_id = (
                f"{source}\0{chunk_start}\0{content}"
            )

            chunk_id = sha256(
                raw_id.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source=source,
                    text=content,
                    start=chunk_start,
                    end=chunk_end,
                )
            )

        if end >= length:
            break

        # Частково повторюємо кінець попереднього
        # фрагмента, щоб зберегти контекст.
        next_start = max(
            start + 1,
            end - overlap_chars,
        )

        # Намагатимемося починати з цілого слова.
        while (
            next_start > start + 1
            and not normalized[next_start - 1].isspace()
        ):
            next_start -= 1

        start = next_start

    return chunks