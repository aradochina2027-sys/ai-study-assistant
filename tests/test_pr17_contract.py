
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def make_app(tmp_path):
    """Окремий тестовий застосунок без Ollama."""

    settings = replace(
        Settings(),
        llm_provider="mock",
        llm_model="mock-study-v1",
        rag_enabled=False,
        db_path=str(tmp_path / "pr17.db"),
        log_file=str(tmp_path / "pr17.jsonl"),
    )

    return create_app(settings)


def test_new_studyai_design(tmp_path):
    """Головна сторінка містить новий дизайн."""

    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    assert "STUDYAI WORKSPACE" in response.text
    assert 'id="question"' in response.text
    assert 'id="sendBtn"' in response.text
    assert 'id="resultSection"' in response.text


def test_input_validation(tmp_path):
    """Некоректні запитання відхиляються."""

    app = make_app(tmp_path)

    with TestClient(app) as client:
        empty = client.post(
            "/api/ask",
            json={"question": "   "},
        )

        long_question = client.post(
            "/api/ask",
            json={"question": "А" * 501},
        )

    assert empty.status_code == 422
    assert long_question.status_code == 422

    for response in (empty, long_question):
        error = response.json()["error"]

        assert error["kind"] == "validation_error"
        assert error["retryable"] is False
        assert error["request_id"]


def test_success_and_saved_result(tmp_path):
    """
    Перевіряє повний API-сценарій:
    створення -> завершення -> читання результату.
    """

    app = make_app(tmp_path)

    def fake_run_job(data, request_id):
        storage = app.state.storage

        storage.set_running(request_id)

        storage.complete(
            request_id,
            {
                "status": "ok",
                "answer": (
                    "git push надсилає коміти "
                    "до віддаленого репозиторію."
                ),
                "sources": ["data/notes.md"],
            },
        )

    app.state.service.run_job = fake_run_job

    with TestClient(app) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "Що робить git push?",
                "documents": ["data/notes.md"],
            },
        )

        assert response.status_code == 202

        request_id = response.json()["request_id"]

        result_response = client.get(
            f"/api/requests/{request_id}"
        )

        assert result_response.status_code == 200

        data = result_response.json()

        assert data["status"] == "completed"
        assert "git push" in data["result"]["answer"]

        assert data["result"]["sources"] == [
            "data/notes.md"
        ]

        # Повторно читаємо збережений результат.
        restored = client.get(
            f"/api/requests/{request_id}"
        )

        assert restored.status_code == 200
        assert restored.json() == data


def test_controlled_provider_error(tmp_path):
    """Помилка передається у форматі API."""

    app = make_app(tmp_path)

    def fake_failed_job(data, request_id):
        storage = app.state.storage

        storage.set_running(request_id)

        storage.fail(
            request_id,
            {
                "kind": "provider_unavailable",
                "message": "AI-провайдер недоступний",
                "retryable": True,
                "status_code": 503,
            },
        )

    app.state.service.run_job = fake_failed_job

    with TestClient(app) as client:
        accepted = client.post(
            "/api/ask",
            json={"question": "Що таке Python?"},
        )

        assert accepted.status_code == 202

        request_id = accepted.json()["request_id"]

        response = client.get(
            f"/api/requests/{request_id}"
        )

    assert response.status_code == 503

    error = response.json()["error"]

    assert error["kind"] == "provider_unavailable"
    assert error["retryable"] is True
    assert error["request_id"] == request_id
    assert "Traceback" not in str(error)


def test_cancel_request(tmp_path):
    """Перевіряє скасування очікувального запиту."""

    app = make_app(tmp_path)

    request_id = "pr17-cancel-check"

    app.state.storage.create(request_id)

    with TestClient(app) as client:
        response = client.post(
            f"/api/requests/{request_id}/cancel"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

        status = client.get(
            f"/api/requests/{request_id}"
        )

        assert status.status_code == 200
        assert status.json()["status"] == "cancelled"

        repeated = client.post(
            f"/api/requests/{request_id}/cancel"
        )

        assert repeated.status_code == 409
        assert repeated.json()["error"]["kind"] == "conflict"
