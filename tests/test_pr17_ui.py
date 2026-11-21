
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def test_pr17_new_design(tmp_path):
    settings = replace(
        Settings(),
        llm_provider="mock",
        rag_enabled=False,
        db_path=str(tmp_path / "ui_test.db"),
        log_file=str(tmp_path / "ui_test.jsonl"),
    )

    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/")

    # Сервер повертає HTML.
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Завжди віддається нова версія інтерфейсу.
    assert (
        response.headers["X-UI-Version"]
        == "StudyAI-PR17"
    )

    # Старий HTML не має використовуватися.
    assert "STUDYAI WORKSPACE" in response.text

    # Перевіряємо елементи нового інтерфейсу.
    assert 'id="question"' in response.text
    assert 'id="sendBtn"' in response.text
    assert 'id="resultSection"' in response.text

    # Браузер не повинен кешувати HTML.
    assert "no-store" in response.headers[
        "Cache-Control"
    ]
