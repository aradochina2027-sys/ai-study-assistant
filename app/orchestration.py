from app.gateway import Gateway
from app.models import (
    AppError,
    ModelOutput,
    ProviderMessage,
    ProviderRequest,
)
from app.prompting import render_prompt, render_question
from app.settings import Settings
from app.telemetry import Telemetry
from app.validation import check_output


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
        self.measurements = {}

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

        if not context.strip():
            raise AppError(
                "empty_documents",
                "Документи порожні",
                400,
            )

        prompt, params = render_prompt(
            self.settings,
            context,
            question,
            documents,
        )

        version = params["prompt_version"]

        self.telemetry.emit(
            request_id,
            "orchestration",
            (
                f"context_v={version};"
                f"sources={','.join(documents)};"
                f"chars={len(context)}"
            ),
        )

        request = ProviderRequest(
            request_id=request_id,
            model=self.settings.llm_model,
            documents=documents,
            timeout_s=self.settings.llm_timeout_s,
            temperature=float(
                params.get("temperature", 0)
            ),
            max_tokens=int(
                params.get("max_tokens", 450)
            ),
            output_schema=ModelOutput.model_json_schema(),
            prompt_version=version,
            messages=[
                ProviderMessage(
                    role="system",
                    content=prompt,
                ),
                ProviderMessage(
                    role="user",
                    content=render_question(question),
                ),
            ],
        )

        # Максимум два виклики:
        # основний і один повтор після помилки.
        for attempt in range(2):
            result = self.gateway.complete(request)

            self.measurements[request_id] = {
                "model": result.model,
                "prompt_version": version,
                "elapsed_ms": result.elapsed_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "usd": (
                    0.0
                    if self.settings.llm_provider
                    in {"mock", "ollama", "recorded"}
                    else None
                ),
                "attempts": attempt + 1,
            }

            try:
                output = check_output(
                    result.text,
                    documents,
                    context,
                )

                self.telemetry.emit(
                    request_id,
                    "orchestration",
                    "validated",
                )

                return output

            except ValueError as error:
                self.telemetry.emit(
                    request_id,
                    "orchestration",
                    f"invalid_output_attempt_{attempt + 1}",
                )

                if attempt == 0:
                    request.messages.append(
                        ProviderMessage(
                            role="user",
                            content=(
                                "Попередня відповідь "
                                "не пройшла перевірку "
                                f"({error}). "
                                "Поверни виправлений JSON."
                            ),
                        )
                    )

        raise AppError(
            "invalid_output",
            "Відповідь AI не пройшла перевірку "
            "навіть після повтору",
            503,
            False,
        )