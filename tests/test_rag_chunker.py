import pytest

from app.rag.chunker import chunk_text


def test_empty_document():
    chunks = chunk_text(
        "",
        source="data/empty.md",
    )

    assert chunks == []


def test_small_document():
    text = "Git commit зберігає зміни."

    chunks = chunk_text(
        text,
        source="data/notes.md",
    )

    assert len(chunks) == 1

    assert chunks[0].text == text
    assert chunks[0].source == "data/notes.md"
    assert chunks[0].chunk_id


def test_large_document():
    text = " ".join(
        f"слово_{i:03d}"
        for i in range(150)
    )

    chunks = chunk_text(
        text,
        source="data/notes.md",
        max_chars=120,
        overlap_chars=25,
    )

    assert len(chunks) > 1

    assert all(
        0 < len(chunk.text) <= 120
        for chunk in chunks
    )

    assert len({
        chunk.chunk_id
        for chunk in chunks
    }) == len(chunks)

    # Сусідні фрагменти перекриваються.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end


def test_invalid_configuration():
    with pytest.raises(ValueError):
        chunk_text(
            "Тест",
            source="data/notes.md",
            max_chars=100,
            overlap_chars=100,
        )