
"""ПР18 — Тести журналювання, ротації та захисту даних."""

import json
import uuid

from app.gateway import Gateway
from app.models import (
    AppError,
    ProviderMessage,
    ProviderRequest,
    ProviderResult,
)
from app.telemetry import Telemetry


def events_in(path):
    """Завантажує JSONL-події."""

    if not path.exists():
        return []

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    return [
        json.loads(line)
        for line in lines
        if line
    ]


def test_jsonl_safe_fields_and_legacy_format(
    tmp_path,
):
    """
    Перевіряє:
    - правильність JSON;
    - сумісність старого формату;
    - дозволені метадані;
    - відсутність секретів.
    """

    path = tmp_path / "app.jsonl"

    telemetry = Telemetry(
        str(path)
    )

    request_id = str(
        uuid.uuid4()
    )

    telemetry.emit(
        request_id,
        "gateway",
        "provider_returned",
        provider="ollama",
        model="qwen3:4b",
        prompt_version="v2",
        schema_version="v1",
        attempt=1,
        duration_ms=12.5,
        input_tokens=10,
        output_tokens=5,
        cached=False,
        fallback=False,
        question="PRIVATE QUESTION",
        api_key="sk-SECRET-123",
        messages=["PRIVATE MESSAGES"],
    )

    telemetry.close()

    rows = events_in(path)

    assert len(rows) == 1

    row = rows[0]

    assert row["request_id"] == request_id
    assert row["layer"] == "gateway"
    assert row["event"] == "provider_returned"

    assert row["prompt_version"] == "v2"
    assert row["input_tokens"] == 10
    assert row["duration_ms"] == 12.5
    assert row["ts"]

    text = path.read_text(
        encoding="utf-8"
    )

    for secret in (
        "PRIVATE QUESTION",
        "PRIVATE MESSAGES",
        "sk-SECRET-123",
    ):
        assert secret not in text

    assert "question" not in row
    assert "api_key" not in row


def test_untrusted_event_request_id_and_model_are_redacted(
    tmp_path,
):
    """
    Недовірені значення не повинні
    потрапляти у файл журналу.
    """

    path = tmp_path / "redacted.jsonl"

    telemetry = Telemetry(
        str(path)
    )

    telemetry.emit(
        "user@example.com SECRET",
        "orchestration",
        "context_v=v2;sources=private_notes.txt;chars=123",
        model="sk-EXPOSE-THIS",
        provider=["ollama"],
        duration_ms=float("nan"),
        customer_email="user@example.com SECRET",
    )

    telemetry.close()

    raw = path.read_text(
        encoding="utf-8"
    )

    assert "user@example.com" not in raw
    assert "private_notes.txt" not in raw
    assert "sk-EXPOSE" not in raw

    row = events_in(path)[0]

    assert row["event"] == "event_rejected"
    assert row["model"] == "other_model"

    assert row["request_id"].startswith(
        "hash_"
    )

    assert "duration_ms" not in row
    assert "provider" not in row


def test_rotating_jsonl_files(tmp_path):
    """
    Коли файл досягає ліміту,
    створюються резервні копії.
    """

    path = tmp_path / "rotating.jsonl"

    telemetry = Telemetry(
        str(path),
        max_bytes=350,
        backup_count=2,
    )

    request_id = str(
        uuid.uuid4()
    )

    for attempt in range(1, 31):
        telemetry.emit(
            request_id,
            "gateway",
            "provider_call",
            attempt=attempt,
        )

    telemetry.close()

    paths = [
        path,
        tmp_path / "rotating.jsonl.1",
        tmp_path / "rotating.jsonl.2",
    ]

    assert paths[1].exists()

    for existing in paths:
        if existing.exists():
            assert events_in(existing)


def test_trace_across_stages(tmp_path):
    """
    Усі етапи одного запиту
    мають той самий request_id.
    """

    path = tmp_path / "trace.jsonl"

    telemetry = Telemetry(
        str(path)
    )

    request_id = str(
        uuid.uuid4()
    )

    for layer, event in (
        ("service", "started"),
        ("rag", "search_started"),
        ("gateway", "provider_call"),
        ("gateway", "provider_returned"),
        ("service", "finished"),
    ):
        telemetry.emit(
            request_id,
            layer,
            event,
        )

    telemetry.close()

    rows = events_in(path)

    assert len(rows) == 5

    assert all(
        row["request_id"] == request_id
        for row in rows
    )

    assert (
        rows[-1]["event"]
        == "finished"
    )


class SequenceProvider:
    """
    Підставний провайдер.

    Повертає результати чи помилки
    в наперед заданій послідовності.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1

        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


def make_request():
    """Створює тестовий запит."""

    return ProviderRequest(
        request_id=str(uuid.uuid4()),
        model="mock-study-v1",
        timeout_s=10.0,
        documents=["data/notes.md"],
        prompt_version="v2",
        messages=[
            ProviderMessage(
                role="user",
                content=(
                    "НЕ ЛОГУВАТИ ЦЕ ЗАПИТАННЯ"
                ),
            )
        ],
    )


def success():
    """Підставна успішна відповідь."""

    return ProviderResult(
        text=(
            '{"status":"ok",'
            '"answer":"private response",'
            '"sources":[]}'
        ),
        model="mock-study-v1",
        input_tokens=12,
        output_tokens=8,
        elapsed_ms=1.0,
    )


def test_gateway_emits_tokens_duration_and_versions(
    tmp_path,
):
    """
    Gateway записує:
    - кількість токенів;
    - тривалість;
    - версію промпту;
    - версію схеми.
    """

    path = tmp_path / "gateway.jsonl"

    telemetry = Telemetry(
        str(path)
    )

    provider = SequenceProvider(
        [success()]
    )

    gateway = Gateway(
        provider,
        telemetry,
        max_attempts=1,
    )

    result = gateway.complete(
        make_request()
    )

    telemetry.close()

    assert result.input_tokens == 12

    rows = events_in(path)

    assert [
        row["event"]
        for row in rows
    ] == [
        "provider_call",
        "provider_returned",
    ]

    returned = next(
        row
        for row in rows
        if row["event"] == "provider_returned"
    )

    assert returned["input_tokens"] == 12
    assert returned["output_tokens"] == 8

    assert returned["duration_ms"] >= 0

    assert returned["prompt_version"] == "v2"
    assert returned["schema_version"] == "v1"
    assert returned["status"] == "ok"

    raw = path.read_text(
        encoding="utf-8"
    )

    assert "НЕ ЛОГУВАТИ" not in raw
    assert "private response" not in raw


def test_gateway_retry_events_hide_error_message(
    tmp_path,
):
    """
    Перевіряє:
    - повтор після тимчасової помилки;
    - дві фактичні спроби;
    - подію retry_succeeded;
    - відсутність тексту винятку.
    """

    path = tmp_path / "retries.jsonl"

    telemetry = Telemetry(
        str(path)
    )

    error = AppError(
        "provider_unavailable",
        "SECRET-EXCEPTION-DETAILS",
        503,
        True,
    )

    provider = SequenceProvider(
        [
            error,
            success(),
        ]
    )

    waits = []

    gateway = Gateway(
        provider,
        telemetry,
        max_attempts=2,
        base_delay_s=0.1,
        sleep_fn=waits.append,
        random_fn=lambda: 1.0,
    )

    gateway.complete(
        make_request()
    )

    telemetry.close()

    assert provider.calls == 2
    assert len(waits) == 1

    rows = events_in(path)

    assert sum(
        row["event"] == "provider_call"
        for row in rows
    ) == 2

    assert any(
        row["event"] == "retry_scheduled"
        for row in rows
    )

    assert any(
        row["event"] == "retry_succeeded"
        for row in rows
    )

    assert any(
        row.get("error_kind")
        == "provider_unavailable"
        for row in rows
    )

    raw = path.read_text(
        encoding="utf-8"
    )

    assert "SECRET-EXCEPTION-DETAILS" not in raw
