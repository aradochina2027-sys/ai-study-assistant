
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import ModelOutput
from app.settings import Settings


def make_settings(tmp_path):
    """
    Безпечна тестова конфігурація.

    Використовується:
    - mock-провайдер;
    - тимчасова SQLite;
    - окремий журнал;
    - вимкнений RAG.
    """

    return replace(
        Settings(),
        llm_provider="mock",
        rag_enabled=False,
        db_path=str(
            tmp_path / "idempotency.db"
        ),
        log_file=str(
            tmp_path / "idempotency.jsonl"
        ),
    )


def install_fake_job(app, calls):
    """
    Замінює AI-генерацію тестовою функцією.

    Це дозволяє перевірити,
    скільки разів фактично
    запускається обробка.
    """

    def fake_run_job(data, request_id):
        calls.append(request_id)

        app.state.storage.set_running(
            request_id
        )

        result = ModelOutput(
            status="ok",
            answer=(
                "git push надсилає коміти "
                "до віддаленого репозиторію."
            ),
            sources=["data/notes.md"],
        )

        app.state.storage.complete(
            request_id,
            result.model_dump(),
        )

    app.state.service.run_job = fake_run_job


def test_duplicate_key_runs_only_once(tmp_path):
    """
    Два однакові запити з одним ключем
    повинні запускати AI лише один раз.
    """

    app = create_app(
        make_settings(tmp_path)
    )

    calls = []
    install_fake_job(app, calls)

    payload = {
        "question": "Що робить git push?",
        "documents": ["data/notes.md"],
    }

    headers = {
        "Idempotency-Key": "pr17-duplicate-001",
    }

    with TestClient(app) as client:
        first = client.post(
            "/api/ask",
            json=payload,
            headers=headers,
        )

        second = client.post(
            "/api/ask",
            json=payload,
            headers=headers,
        )

        assert first.status_code == 202
        assert second.status_code == 202

        first_id = first.json()["request_id"]
        second_id = second.json()["request_id"]

        assert first_id == second_id

        status = client.get(
            f"/api/requests/{first_id}"
        )

        assert status.status_code == 200
        assert status.json()["status"] == "completed"

    # AI-завдання запущено тільки один раз.
    assert calls == [first_id]


def test_same_key_different_question_conflict(tmp_path):
    """
    Один ключ не можна використати
    для двох різних запитань.
    """

    app = create_app(
        make_settings(tmp_path)
    )

    calls = []
    install_fake_job(app, calls)

    headers = {
        "Idempotency-Key": "pr17-conflict-001",
    }

    with TestClient(app) as client:
        first = client.post(
            "/api/ask",
            json={
                "question": "Що таке Git?",
            },
            headers=headers,
        )

        second = client.post(
            "/api/ask",
            json={
                "question": "Що таке Python?",
            },
            headers=headers,
        )

    assert first.status_code == 202
    assert second.status_code == 409

    error = second.json()["error"]

    assert error["kind"] == "idempotency_conflict"
    assert error["retryable"] is False

    # Другий запуск не відбувся.
    assert len(calls) == 1


def test_key_survives_server_restart(tmp_path):
    """
    Повторення запиту після перезапуску
    сервера використовує старий request_id.
    """

    settings = make_settings(tmp_path)

    payload = {
        "question": "Що таке SQL?",
        "documents": ["data/notes.md"],
    }

    headers = {
        "Idempotency-Key": "pr17-restart-001",
    }

    first_app = create_app(settings)

    first_calls = []
    install_fake_job(
        first_app,
        first_calls,
    )

    with TestClient(first_app) as client:
        first = client.post(
            "/api/ask",
            json=payload,
            headers=headers,
        )

    assert first.status_code == 202

    original_id = first.json()["request_id"]

    # Створюємо новий застосунок із тією
    # самою базою SQLite.
    restarted_app = create_app(settings)

    restarted_calls = []
    install_fake_job(
        restarted_app,
        restarted_calls,
    )

    with TestClient(restarted_app) as client:
        repeated = client.post(
            "/api/ask",
            json=payload,
            headers=headers,
        )

        result = client.get(
            f"/api/requests/{original_id}"
        )

    assert repeated.status_code == 202
    assert repeated.json()["request_id"] == original_id

    assert result.status_code == 200
    assert result.json()["status"] == "completed"

    # Після перезапуску нового виклику AI немає.
    assert restarted_calls == []


def test_invalid_idempotency_key(tmp_path):
    """
    Перевіряє неправильний формат ключа.
    """

    app = create_app(
        make_settings(tmp_path)
    )

    calls = []
    install_fake_job(app, calls)

    with TestClient(app) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "Що таке Git?",
            },
            headers={
                "Idempotency-Key": "bad",
            },
        )

    assert response.status_code == 400

    error = response.json()["error"]

    assert error["kind"] == "invalid_idempotency_key"
    assert error["retryable"] is False

    # Жодна генерація не запускалася.
    assert calls == []


def test_requests_without_key_still_work(tmp_path):
    """
    Зворотна сумісність:
    старі клієнти можуть працювати
    без Idempotency-Key.
    """

    app = create_app(
        make_settings(tmp_path)
    )

    calls = []
    install_fake_job(app, calls)

    payload = {
        "question": "Що таке Git?",
    }

    with TestClient(app) as client:
        first = client.post(
            "/api/ask",
            json=payload,
        )

        second = client.post(
            "/api/ask",
            json=payload,
        )

    assert first.status_code == 202
    assert second.status_code == 202

    assert (
        first.json()["request_id"]
        != second.json()["request_id"]
    )

    # Без ключа дозволені незалежні запити.
    assert len(calls) == 2
