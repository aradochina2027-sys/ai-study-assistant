
import json

import httpx

from app.rag.index import build_index, load_index


def make_client(counter):
    def handler(request):
        assert request.url.path == "/api/embed"

        counter["calls"] += 1

        payload = json.loads(request.content)
        texts = payload["input"]

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.6, 0.8]
                    for _ in texts
                ]
            },
        )

    return httpx.Client(
        transport=httpx.MockTransport(handler)
    )


def test_build_index(tmp_path):
    document = tmp_path / "notes.md"
    index_path = tmp_path / "index.json"

    document.write_text(
        "git commit зберігає зміни.",
        encoding="utf-8",
    )

    counter = {"calls": 0}

    with make_client(counter) as client:
        result = build_index(
            document,
            index_path,
            client=client,
        )

    assert index_path.exists()
    assert result["embedding_model"] == "qwen3-embedding:0.6b"
    assert result["dimensions"] == 2
    assert len(result["chunks"]) == 1
    assert counter["calls"] == 1

    loaded = load_index(index_path)

    assert loaded == result


def test_index_reused_without_new_embedding(tmp_path):
    document = tmp_path / "notes.md"
    index_path = tmp_path / "index.json"

    document.write_text(
        "Git — система контролю версій.",
        encoding="utf-8",
    )

    counter = {"calls": 0}

    with make_client(counter) as client:
        first = build_index(
            document,
            index_path,
            client=client,
        )

        second = build_index(
            document,
            index_path,
            client=client,
        )

    assert first == second

    # Другий виклик не звертався до Ollama.
    assert counter["calls"] == 1


def test_document_change_rebuilds_index(tmp_path):
    document = tmp_path / "notes.md"
    index_path = tmp_path / "index.json"

    document.write_text(
        "Перша версія документа.",
        encoding="utf-8",
    )

    counter = {"calls": 0}

    with make_client(counter) as client:
        first = build_index(
            document,
            index_path,
            client=client,
        )

        document.write_text(
            "Друга версія документа.",
            encoding="utf-8",
        )

        second = build_index(
            document,
            index_path,
            client=client,
        )

    assert first["document_sha256"] != second["document_sha256"]
    assert counter["calls"] == 2


def test_empty_document(tmp_path):
    document = tmp_path / "empty.md"
    index_path = tmp_path / "index.json"

    document.write_text("", encoding="utf-8")

    counter = {"calls": 0}

    with make_client(counter) as client:
        result = build_index(
            document,
            index_path,
            client=client,
        )

    assert result["chunks"] == []
    assert result["dimensions"] == 0
    assert counter["calls"] == 0
