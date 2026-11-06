import hashlib
import json
import time

from collections.abc import Iterator

from app.models import AppError, ProviderRequest
from app.providers.base import BaseProvider
from app.settings import Settings
from app.telemetry import Telemetry


class MockProvider(BaseProvider):
    def __init__(
        self,
        settings: Settings,
        telemetry: Telemetry,
    ):
        self.settings = settings
        self.telemetry = telemetry

    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:

        self.telemetry.emit(
            request.request_id,
            "provider",
            "mock_called",
        )

        if (
            self.settings.mock_delay_ms
            >= request.timeout_s * 1000
        ):
            raise AppError(
                "timeout",
                "Перевищено час відповіді моделі",
                504,
                True,
            )

        if self.settings.mock_delay_ms:
            time.sleep(
                self.settings.mock_delay_ms / 1000
            )

        ratio = int.from_bytes(
            hashlib.sha256(
                request.request_id.encode()
            ).digest()[:4],
            "big",
        ) / 2**32

        failure = (
            self.settings.mock_failure_kind
            if ratio < self.settings.mock_failure_rate
            else "none"
        )

        errors = {
            "timeout": (
                "timeout",
                "Модель не відповіла вчасно",
                504,
            ),
            "rate_limit": (
                "rate_limit",
                "Ліміт запитів провайдера",
                429,
            ),
            "unavailable": (
                "provider_unavailable",
                "Провайдер тимчасово недоступний",
                503,
            ),
        }

        if failure in errors:
            kind, message, status = errors[failure]

            raise AppError(
                kind, message, status, True
            )

        if failure == "invalid_json":
            yield "{invalid json"
            return

        question = request.messages[-1].content.lower()

        pairs = [
            (
                ["git init", "створити", "репозитор"],
                "git init — створює Git-репозиторій.",
            ),
            (
                ["git status", "стан файлів", "гіт статус"],
                "git status — показує стан файлів.",
            ),
            (
                ["git commit", "коміт"],
                "git commit — зберігає зміни в історії.",
            ),
            (
                ["git push"],
                "git push — надсилає коміти "
                "у віддалений репозиторій.",
            ),
            (
                ["словник", "ключ-значення", "dict"],
                "dict — колекція пар ключ-значення.",
            ),
            (
                ["список", "list"],
                "list — впорядкована "
                "змінювана колекція.",
            ),
            (
                ["представлення", "create view"],
                "CREATE VIEW — створює представлення SQL.",
            ),
        ]

        answer = next(
            (
                value
                for terms, value in pairs
                if any(term in question for term in terms)
            ),
            None,
        )

        if answer:
            payload = {
                "status": "ok",
                "answer": answer,
                "sources": request.documents,
            }
        else:
            payload = {
                "status": "no_evidence",
                "answer": (
                    "У матеріалах немає "
                    "підтвердженої відповіді."
                ),
                "sources": [],
            }

        content = json.dumps(
            payload, ensure_ascii=False
        )

        if failure == "mid_stream":
            yield content[:12]

            raise AppError(
                "mid_stream",
                "Потік відповіді перервано",
                503,
                True,
            )

        for offset in range(0, len(content), 60):
            yield content[offset:offset + 60]