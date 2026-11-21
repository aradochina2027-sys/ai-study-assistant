
"""
ПР18 — Тести лімітів та стану надійності API.

Перевіряємо:
1. Стан Circuit Breaker у health.
2. Degraded при відкритому запобіжнику.
3. HTTP 429 і Retry-After.
4. Незалежні ліміти клієнтів.
5. Ідемпотентність ПР17.
6. Retry-After зі збереженої помилки.
7. Retry-After для Circuit Breaker.
8. Збереження формату 422.
9. Некоректний Idempotency-Key.

Справжня Ollama не використовується.
"""

import sqlite3
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import AppError
from app.settings import Settings


def make_app(tmp_path):
    """
    Створює ізольований тестовий застосунок.

    Використовує:
    - mock-провайдер;
    - тимчасову базу SQLite;
    - тимчасовий журнал;
    - вимкнений RAG.
    """

    settings = replace(
        Settings(),
        llm_provider="mock",
        rag_enabled=False,
        db_path=str(
            tmp_path / "pr18_api.db"
        ),
        log_file=str(
            tmp_path / "pr18_api.jsonl"
        ),
    )

    app = create_app(settings)

    # Не запускаємо модель у тестах.
    # Лише перевіряємо HTTP-контракт.
    app.state.service.run_job = (
        lambda data, request_id: None
    )

    return app


def ask_payload(question="Що таке Python?"):
    return {
        "question": question,
        "documents": ["data/notes.md"],
    }


def open_breaker(app):
    """
    Штучно створює три збої доступності.

    Зовнішніх HTTP-викликів немає.
    """

    breaker = app.state.gateway.breaker

    error = AppError(
        "provider_unavailable",
        "Провайдер недоступний",
        503,
        True,
    )

    for _ in range(3):
        breaker.record_failure(error)


def test_health_reports_breaker_and_limits(
    tmp_path,
):
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/api/health"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["breaker"]["state"] == "closed"

    assert "limits" in data

    assert (
        data["limits"][
            "client_requests_per_window"
        ] >= 1
    )

    assert data["limits"]["max_concurrent"] >= 1


def test_health_degraded_when_breaker_open(
    tmp_path,
):
    app = make_app(tmp_path)

    open_breaker(app)

    with TestClient(app) as client:
        response = client.get(
            "/api/health"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"
    assert data["breaker"]["state"] == "open"

    assert data["breaker"]["retry_after_s"] > 0


def test_rate_limit_returns_429_with_retry_after(
    tmp_path,
    monkeypatch,
):
    """
    Дозволяємо два запити за 60 секунд.

    Третій повинен отримати 429.
    """

    monkeypatch.setenv(
        "STUDYAI_RATE_LIMIT_MAX",
        "2",
    )

    monkeypatch.setenv(
        "STUDYAI_RATE_LIMIT_WINDOW_S",
        "60",
    )

    app = make_app(tmp_path)

    with TestClient(app) as client:
        first = client.post(
            "/api/ask",
            json=ask_payload("Що таке Git?"),
        )

        second = client.post(
            "/api/ask",
            json=ask_payload("Що таке SQL?"),
        )

        third = client.post(
            "/api/ask",
            json=ask_payload("Що таке Python?"),
        )

    assert first.status_code == 202
    assert second.status_code == 202
    assert third.status_code == 429

    error = third.json()["error"]

    assert error["kind"] == "client_rate_limit"
    assert error["retryable"] is True
    assert error["request_id"]

    assert third.headers["X-Request-ID"] == (
        error["request_id"]
    )

    retry_after = int(
        third.headers["Retry-After"]
    )

    assert 1 <= retry_after <= 60

    # Відхилений запит не створюється
    # у SQLite.
    with sqlite3.connect(
        app.state.storage.path
    ) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM requests"
        ).fetchone()[0]

    assert count == 2


def test_rate_limit_is_per_client(
    tmp_path,
    monkeypatch,
):
    """
    Ліміт одного клієнта не повинен
    блокувати інший ідентифікатор.
    """

    monkeypatch.setenv(
        "STUDYAI_RATE_LIMIT_MAX",
        "1",
    )

    app = make_app(tmp_path)

    limiter = app.state.rate_limiter

    limiter.check("client-A")

    assert limiter.remaining("client-A") == 0
    assert limiter.remaining("client-B") == 1

    limiter.check("client-B")

    assert limiter.remaining("client-B") == 0


def test_idempotent_request_does_not_launch_twice(
    tmp_path,
):
    """
    Однаковий ключ і однаковий запит
    не створюють двох фонових завдань.
    """

    app = make_app(tmp_path)

    calls = []

    def fake_job(data, request_id):
        calls.append(request_id)

    app.state.service.run_job = fake_job

    headers = {
        "Idempotency-Key": "pr18-unique-key-001",
    }

    payload = ask_payload(
        "Поясни команду git push"
    )

    with TestClient(app) as client:
        first = client.post(
            "/api/ask",
            headers=headers,
            json=payload,
        )

        second = client.post(
            "/api/ask",
            headers=headers,
            json=payload,
        )

    assert first.status_code == 202
    assert second.status_code == 202

    first_id = first.json()["request_id"]
    second_id = second.json()["request_id"]

    assert first_id == second_id
    assert calls == [first_id]

    with sqlite3.connect(
        app.state.storage.path
    ) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM requests"
        ).fetchone()[0]

    assert count == 1


def test_saved_error_preserves_retry_after(
    tmp_path,
):
    """
    Збережений збій із retry_after_s
    повертає HTTP-заголовок Retry-After.
    """

    app = make_app(tmp_path)

    request_id = "pr18-saved-rate-limit"

    storage = app.state.storage

    storage.create(request_id)
    storage.set_running(request_id)

    storage.fail(
        request_id,
        {
            "kind": "rate_limit",
            "message": (
                "Провайдер тимчасово обмежив запити."
            ),
            "status_code": 429,
            "retryable": True,
            "retry_after_s": 2.2,
        },
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/requests/{request_id}"
        )

    assert response.status_code == 429

    assert response.headers["Retry-After"] == "3"
    assert (
        response.json()["error"]["kind"]
        == "rate_limit"
    )


def test_open_breaker_error_has_retry_after(
    tmp_path,
):
    """
    Навіть без збереженого Retry-After
    сервер бере актуальний час із breaker.
    """

    app = make_app(tmp_path)

    open_breaker(app)

    request_id = "pr18-breaker-failure"

    storage = app.state.storage

    storage.create(request_id)
    storage.set_running(request_id)

    storage.fail(
        request_id,
        {
            "kind": "circuit_open",
            "message": (
                "Провайдер тимчасово недоступний."
            ),
            "status_code": 503,
            "retryable": True,
        },
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/requests/{request_id}"
        )

    assert response.status_code == 503

    assert (
        response.json()["error"]["kind"]
        == "circuit_open"
    )

    assert int(
        response.headers["Retry-After"]
    ) >= 1


def test_validation_still_has_field_errors(
    tmp_path,
):
    """
    Переконуємося, що зміни ПР18
    не зламали валідацію ПР17.
    """

    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/ask",
            json={
                "question": "   ",
            },
        )

    assert response.status_code == 422

    data = response.json()

    assert data["error"]["kind"] == (
        "validation_error"
    )

    assert any(
        item["field"] == "question"
        for item in data["error"]["fields"]
    )

    assert response.headers["X-Request-ID"]


def test_invalid_idempotency_key_is_400(
    tmp_path,
):
    """
    Некоректний ключ відхиляється
    до створення завдання.
    """

    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/ask",
            json=ask_payload(),
            headers={
                "Idempotency-Key": "bad",
            },
        )

    assert response.status_code == 400

    error = response.json()["error"]

    assert error["kind"] == (
        "invalid_idempotency_key"
    )

    with sqlite3.connect(
        app.state.storage.path
    ) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM requests"
        ).fetchone()[0]

    assert count == 0
