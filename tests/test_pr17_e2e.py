
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def make_settings(tmp_path, **changes):
    """
    Окрема конфігурація для тестів ПР17.

    Не використовує Ollama і не змінює
    основну базу даних проєкту.
    """

    settings = replace(
        Settings(),
        llm_provider="mock",
        llm_model="mock-study-v1",
        rag_enabled=False,
        db_path=str(tmp_path / "pr17_test.db"),
        log_file=str(tmp_path / "pr17_test.jsonl"),
        mock_failure_rate=0.0,
    )

    return replace(settings, **changes)


def test_pr17_page_and_openapi(tmp_path):
    """
    Перевірка головної сторінки
    та правил введення.
    """

    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        page = client.get("/")

        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]

        assert 'id="question"' in page.text

        assert (
            "StudyAI" in page.text
            or "AI Study Assistant" in page.text
        )

        schema_response = client.get("/openapi.json")

        assert schema_response.status_code == 200

        schema = schema_response.json()

        question_field = (
            schema["components"]["schemas"]
            ["AskInput"]["properties"]["question"]
        )

        assert question_field["minLength"] == 1
        assert question_field["maxLength"] == 500


def test_pr17_full_success_and_restore(tmp_path):
    """
    Повний успішний сценарій:

    1. Створення запиту.
    2. Отримання результату.
    3. Перевірка джерела.
    4. Повторне відкриття результату.
    """

    settings = make_settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        accepted = client.post(
            "/api/ask",
            json={
                "question": "Що робить git push?",
                "documents": ["data/notes.md"],
            },
        )

        assert accepted.status_code == 202

        accepted_data = accepted.json()

        assert accepted_data["status"] == "pending"

        request_id = accepted_data["request_id"]

        assert request_id

        response = client.get(
            f"/api/requests/{request_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "completed"

        result = data["result"]

        assert result["status"] == "ok"
        assert "git push" in result["answer"]

        assert result["sources"] == [
            "data/notes.md"
        ]

    # Імітуємо повторне відкриття застосунку.
    # Нова програма читає попередній результат
    # із тієї самої тестової бази SQLite.

    restored_app = create_app(settings)

    with TestClient(restored_app) as client:
        restored = client.get(
            f"/api/requests/{request_id}"
        )

        assert restored.status_code == 200

        restored_data = restored.json()

        assert restored_data["status"] == "completed"
        assert restored_data["result"] == result


def test_pr17_validation_errors(tmp_path):
    """
    Перевірка порожнього і задовгого запитання.
    """

    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        empty = client.post(
            "/api/ask",
            json={"question": "   "},
        )

        assert empty.status_code == 422

        empty_error = empty.json()["error"]

        assert empty_error["kind"] == "validation_error"
        assert empty_error["retryable"] is False
        assert empty_error["request_id"]

        long_question = client.post(
            "/api/ask",
            json={"question": "А" * 501},
        )

        assert long_question.status_code == 422

        long_error = long_question.json()["error"]

        assert long_error["kind"] == "validation_error"


def test_pr17_provider_failure(tmp_path):
    """
    Перевірка контрольованої помилки
    недоступного AI-провайдера.
    """

    settings = make_settings(
        tmp_path,
        mock_failure_rate=1.0,
        mock_failure_kind="unavailable",
    )

    app = create_app(settings)

    with TestClient(app) as client:
        accepted = client.post(
            "/api/ask",
            json={
                "question": "Що робить git push?",
            },
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

        # Внутрішні подробиці не мають
        # потрапляти у відповідь користувачу.

        assert "Traceback" not in str(error)


def test_pr17_cancel_pending_request(tmp_path):
    """
    Перевірка скасування завдання,
    яке ще перебуває у стані pending.
    """

    app = create_app(make_settings(tmp_path))

    request_id = "pr17-cancel-test"

    # Створюємо очікувальне завдання
    # без запуску генерації.
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

        # Повторне скасування вже скасованого
        # завдання повинно повернути конфлікт.
        repeat = client.post(
            f"/api/requests/{request_id}/cancel"
        )

        assert repeat.status_code == 409

        error = repeat.json()["error"]

        assert error["kind"] == "conflict"
