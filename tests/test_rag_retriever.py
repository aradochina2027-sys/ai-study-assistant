
import json

import httpx
import pytest

from app.rag.retriever import (
    cosine_similarity,
    search_index,
)


def make_index(tmp_path, chunks=None):
    if chunks is None:
        chunks = [
            {
                "chunk_id": "git",
                "source": "data/notes.md",
                "text": "git commit зберігає зміни.",
                "start": 0,
                "end": 28,
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "python",
                "source": "data/notes.md",
                "text": "list — список у Python.",
                "start": 30,
                "end": 55,
                "embedding": [0.0, 1.0],
            },
            {
                "chunk_id": "repository",
                "source": "data/notes.md",
                "text": "Git зберігає історію проєкту.",
                "start": 60,
                "end": 90,
                "embedding": [0.8, 0.6],
            },
        ]

    path = tmp_path / "index.json"

    data = {
        "schema_version": 1,
        "embedding_model": "qwen3-embedding:0.6b",
        "dimensions": 2 if chunks else 0,
        "chunks": chunks,
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


def make_client(vector):
    def handler(request):
        assert request.url.path == "/api/embed"

        payload = json.loads(request.content)

        assert payload["model"] == "qwen3-embedding:0.6b"
        assert len(payload["input"]) == 1

        return httpx.Response(
            200,
            json={"embeddings": [vector]},
        )

    return httpx.Client(
        transport=httpx.MockTransport(handler)
    )


def test_cosine_similarity():
    assert cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    ) == pytest.approx(1.0)

    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_search_returns_relevant_chunks(tmp_path):
    index_path = make_index(tmp_path)

    with make_client([1.0, 0.0]) as client:
        hits = search_index(
            "Як зберегти зміни?",
            index_path=index_path,
            top_k=2,
            min_score=0.1,
            client=client,
        )

    assert len(hits) == 2
    assert hits[0].chunk_id == "git"
    assert hits[1].chunk_id == "repository"

    assert hits[0].score > hits[1].score
    assert hits[0].source == "data/notes.md"


def test_min_score_filters_results(tmp_path):
    index_path = make_index(tmp_path)

    with make_client([1.0, 0.0]) as client:
        hits = search_index(
            "Git",
            index_path=index_path,
            top_k=3,
            min_score=0.9,
            client=client,
        )

    assert len(hits) == 1
    assert hits[0].chunk_id == "git"


def test_empty_index(tmp_path):
    index_path = make_index(tmp_path, chunks=[])

    hits = search_index(
        "Тест",
        index_path=index_path,
    )

    assert hits == []


def test_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        search_index(
            "Тест",
            index_path=tmp_path / "missing.json",
        )


def test_invalid_input(tmp_path):
    index_path = make_index(tmp_path)

    with pytest.raises(ValueError):
        search_index(
            "",
            index_path=index_path,
        )

    with pytest.raises(ValueError):
        search_index(
            "Тест",
            index_path=index_path,
            top_k=0,
        )

    with pytest.raises(ValueError):
        search_index(
            "Тест",
            index_path=index_path,
            min_score=2.0,
        )


def test_embedding_dimension_mismatch(tmp_path):
    index_path = make_index(tmp_path)

    with make_client([1.0, 0.0, 0.0]) as client:
        with pytest.raises(ValueError):
            search_index(
                "Git",
                index_path=index_path,
                client=client,
            )
