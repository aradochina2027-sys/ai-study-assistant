
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.embedder import EmbeddingError
from app.rag.retriever import SearchHit
from app.settings import Settings


def make_app(tmp_path):
    settings = replace(
        Settings(),
        db_path=str(tmp_path / "rag_test.db"),
        log_file=str(tmp_path / "rag_test.jsonl"),
        rag_enabled=True,
    )

    return create_app(settings)


def test_rag_success(tmp_path, monkeypatch):
    calls = {"index": 0, "search": 0}

    def fake_build_index(*args, **kwargs):
        calls["index"] += 1
        return {"chunks": []}

    def fake_search(question, **kwargs):
        calls["search"] += 1

        return [
            SearchHit(
                chunk_id="git-push",
                source="data/notes.md",
                text=(
                    "git push — надсилає коміти "
                    "у віддалений репозиторій."
                ),
                start=0,
                end=60,
                score=0.95,
            )
        ]

    monkeypatch.setattr(
        "app.service.build_index",
        fake_build_index,
    )

    monkeypatch.setattr(
        "app.service.search_index",
        fake_search,
    )

    api = make_app(tmp_path)

    with TestClient(api) as client:
        accepted = client.post(
            "/api/ask",
            json={"question": "Що робить git push?"},
        )

        assert accepted.status_code == 202

        response = client.get(
            accepted.json()["status_url"]
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["result"]["status"] == "ok"
    assert "git push" in data["result"]["answer"]
    assert data["result"]["sources"] == [
        "data/notes.md"
    ]

    assert calls["index"] == 1
    assert calls["search"] == 1


def test_rag_no_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.service.build_index",
        lambda *args, **kwargs: {"chunks": []},
    )

    monkeypatch.setattr(
        "app.service.search_index",
        lambda *args, **kwargs: [],
    )

    api = make_app(tmp_path)

    with TestClient(api) as client:
        accepted = client.post(
            "/api/ask",
            json={
                "question": "Невідоме запитання"
            },
        )

        response = client.get(
            accepted.json()["status_url"]
        )

    assert response.status_code == 200

    result = response.json()["result"]

    assert result["status"] == "no_evidence"
    assert result["sources"] == []


def test_rag_unavailable(tmp_path, monkeypatch):
    def failed_index(*args, **kwargs):
        raise EmbeddingError("test provider failure")

    monkeypatch.setattr(
        "app.service.build_index",
        failed_index,
    )

    api = make_app(tmp_path)

    with TestClient(api) as client:
        accepted = client.post(
            "/api/ask",
            json={"question": "Що таке Git?"},
        )

        response = client.get(
            accepted.json()["status_url"]
        )

    assert response.status_code == 503

    error = response.json()["error"]

    assert error["kind"] == "rag_unavailable"
    assert error["retryable"] is True

    # Внутрішні подробиці не потрапляють
    # у відповідь користувачеві.
    assert "test provider failure" not in str(error)
