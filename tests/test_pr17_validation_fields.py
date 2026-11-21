
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def make_app(tmp_path):
    """
    Тестовий FastAPI-застосунок.

    Використовує mock-провайдер
    і тимчасову базу SQLite.
    """

    settings = replace(
        Settings(),
        llm_provider="mock",
        rag_enabled=False,
        db_path=str(
            tmp_path / "validation.db"
        ),
        log_file=str(
            tmp_path / "validation.jsonl"
        ),
    )

    return create_app(settings)


def test_empty_question_field_error(tmp_path):
    """
    Перевірка порожнього запитання.
    """

    api = make_app(tmp_path)

    with TestClient(api) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "   ",
            },
        )

    assert response.status_code == 422

    data = response.json()
    error = data["error"]

    assert error["kind"] == "validation_error"
    assert error["retryable"] is False
    assert error["request_id"]

    assert any(
        item["field"] == "question"
        for item in error["fields"]
    )


def test_long_question_field_error(tmp_path):
    """
    Перевірка обмеження в 500 символів.
    """

    api = make_app(tmp_path)

    with TestClient(api) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "А" * 501,
            },
        )

    assert response.status_code == 422

    error = response.json()["error"]

    assert any(
        item["field"] == "question"
        for item in error["fields"]
    )


def test_documents_field_error(tmp_path):
    """
    Перевірка некоректного списку документів.
    """

    api = make_app(tmp_path)

    with TestClient(api) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "Що таке Python?",
                "documents": [],
            },
        )

    assert response.status_code == 422

    error = response.json()["error"]

    assert any(
        item["field"] == "documents"
        for item in error["fields"]
    )


def test_compatibility_with_browser(tmp_path):
    """
    Старий JavaScript очікує поле detail.

    Перевіряємо, що воно присутнє
    разом із новим error.fields.
    """

    api = make_app(tmp_path)

    with TestClient(api) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "",
            },
        )

    assert response.status_code == 422

    data = response.json()

    assert "error" in data
    assert "detail" in data

    detail = data["detail"]

    assert isinstance(detail, list)

    assert any(
        "question" in item["loc"]
        for item in detail
    )

    assert all(
        item.get("msg")
        for item in detail
    )


def test_old_error_format_still_works(tmp_path):
    """
    Звичайні помилки не повинні вимагати
    додаткових полів валідації.
    """

    api = make_app(tmp_path)

    with TestClient(api) as client:
        response = client.get(
            "/api/requests/not-existing-request"
        )

    assert response.status_code == 404

    data = response.json()
    error = data["error"]

    assert error["kind"] == "not_found"
    assert error["request_id"]
    assert error["retryable"] is False

    assert "fields" not in error
    assert "detail" not in data
