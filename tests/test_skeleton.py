
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import AskInput
from app.settings import Settings


# Створення тестового застосунку
def make_app(tmp_path, **changes):
    settings = replace(
        Settings(),
        db_path=str(tmp_path / "test.db"),
        log_file=str(tmp_path / "test.jsonl"),
        **changes,
    )

    return create_app(settings)


# ТЕСТ 1. Перевірка всіх рівнів застосунку
def test_api_through_all_layers(tmp_path):
    api = make_app(tmp_path)

    with TestClient(api) as client:

        # Перевірка стану сервера
        health = client.get("/api/health")

        assert health.status_code == 200
        assert health.json()["mode"] == "mock"

        # Надсилання запитання
        response = client.post(
            "/api/ask",
            json={
                "question": "Як створити Git-репозиторій?"
            },
        )

        assert response.status_code == 202

        request_id = response.json()["request_id"]

        # Отримання результату
        result = client.get(
            response.json()["status_url"]
        )

        assert result.status_code == 200
        assert result.json()["status"] == "completed"

        assert (
            "git init"
            in result.json()["result"]["answer"]
        )

        assert result.json()["result"]["sources"] == [
            "data/notes.md"
        ]

    # Перевірка журналювання всіх рівнів
    log_file = tmp_path / "test.jsonl"

    lines = [
        json.loads(line)
        for line in log_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    layers = {
        line["layer"]
        for line in lines
        if line["request_id"] == request_id
    }

    expected_layers = {
        "controller",
        "service",
        "storage",
        "orchestration",
        "gateway",
        "provider",
    }

    assert expected_layers <= layers


# ТЕСТ 2. Перевірка сервісу без HTTP
def test_service_without_http(tmp_path):
    api = make_app(tmp_path)

    result = api.state.service.handle_question(
        AskInput(
            question="Що робить git push?"
        ),
        "direct-test",
    )

    assert result.status == "ok"
    assert "git push" in result.answer


# ТЕСТ 3. Контрольована помилка провайдера
def test_failure_is_controlled(tmp_path):
    api = make_app(
        tmp_path,
        mock_failure_rate=1.0,
        mock_failure_kind="unavailable",
    )

    with TestClient(api) as client:

        accepted = client.post(
            "/api/ask",
            json={
                "question": "Що таке Git?"
            },
        )

        assert accepted.status_code == 202

        request_id = accepted.json()["request_id"]

        failed = client.get(
            accepted.json()["status_url"]
        )

        assert failed.status_code == 503

        assert failed.json()["error"] == {
            "kind": "provider_unavailable",
            "message": "Провайдер тимчасово недоступний",
            "retryable": True,
            "request_id": request_id,
        }


# ТЕСТ 4. Контрольована помилка JSON
def test_invalid_json_is_controlled(tmp_path):
    api = make_app(
        tmp_path,
        mock_failure_rate=1.0,
        mock_failure_kind="invalid_json",
    )

    with TestClient(api) as client:

        accepted = client.post(
            "/api/ask",
            json={
                "question": "Git status"
            },
        )

        assert accepted.status_code == 202

        failed = client.get(
            accepted.json()["status_url"]
        )

        assert failed.status_code == 503

        # Оновлена назва помилки для ПР15
        assert (
            failed.json()["error"]["kind"]
            == "invalid_output"
        )
