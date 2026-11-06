from pydantic import ValidationError

from app.gateway import Gateway
from app.models import (
    AppError,
    ModelOutput,
    ProviderMessage,
    ProviderRequest,
)
from app.settings import Settings
from app.telemetry import Telemetry


class Orchestrator:
    def __init__(
        self,
        gateway: Gateway,
        settings: Settings,
        telemetry: Telemetry,
    ):
        self.gateway = gateway
        self.settings = settings
        self.telemetry = telemetry

    def answer(
        self,
        question: str,
        documents: list[str],
        context: str,
        request_id: str,
    ) -> ModelOutput:

        self.telemetry.emit(
            request_id,
            "orchestration",
            "started",
        )

        request = ProviderRequest(
            request_id=request_id,
            model=self.settings.llm_model,
            documents=documents,
            timeout_s=self.settings.llm_timeout_s,
            messages=[
                ProviderMessage(
                    role="system",
                    content=(
                        "Відповідай лише за "
                        "наданими джерелами."
                    ),
                ),
                ProviderMessage(
                    role="user",
                    content=question,
                ),
            ],
        )

        # У ПР15 контекст буде передано
        # у справжню модель.

        if not context.strip():
            raise AppError(
                "empty_documents",
                "Документи порожні",
                400,
            )

        result = self.gateway.complete(request)

        try:
            output = ModelOutput.model_validate_json(
                result.text
            )

        except ValidationError as exc:
            raise AppError(
                "invalid_model_output",
                "Модель повернула невалідний JSON",
                503,
                True,
            ) from exc

        if any(
            source not in documents
            for source in output.sources
        ):
            raise AppError(
                "invalid_sources",
                "Модель послалася "
                "на невідоме джерело",
                503,
            )

        if (
            output.status == "ok"
            and not output.sources
        ):
            raise AppError(
                "missing_sources",
                "Відповідь не містить джерел",
                503,
            )

        self.telemetry.emit(
            request_id,
            "orchestration",
            "validated",
        )

        return output