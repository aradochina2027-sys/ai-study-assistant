import json

from collections.abc import Iterator
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from app.models import (
    AppError,
    ProviderRequest,
    ProviderResult,
)
from app.providers.base import BaseProvider
from app.settings import Settings


def recording_key(request: ProviderRequest) -> str:
    info = request.model_dump(
        exclude={
            "request_id",
            "timeout_s",
            "documents",
        }
    )

    raw = json.dumps(
        info,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return sha256(raw).hexdigest()


class RecordedProvider(BaseProvider):
    def __init__(
        self,
        settings: Settings,
        upstream: BaseProvider | None = None,
    ):
        self.settings = settings
        self.upstream = upstream
        self.root = Path(settings.recordings_dir)

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:

        path = self.root / (
            recording_key(request) + ".json"
        )

        if self.settings.record_mode == "replay":

            if not path.is_file():
                raise AppError(
                    "recording_missing",
                    "Для запиту немає записаної відповіді",
                    404,
                )

            try:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )

                return ProviderResult.model_validate(
                    data["result"]
                )

            except Exception as error:
                raise AppError(
                    "recording_invalid",
                    "Запис відповіді пошкоджений",
                    503,
                ) from error

        if self.upstream is None:
            raise AppError(
                "provider_unavailable",
                "Немає провайдера для запису",
                503,
            )

        response = self.upstream.complete(request)

        if self.settings.record_mode == "record":
            self.root.mkdir(
                parents=True, exist_ok=True
            )

            if not path.exists():
                data = {
                    "model": request.model,
                    "prompt_version": request.prompt_version,
                    "recorded_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "request_hash": recording_key(request),
                    "result": response.model_dump(),
                }

                path.write_text(
                    json.dumps(
                        data,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        return response

    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:
        yield self.complete(request).text