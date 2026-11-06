import uuid

from app.models import AppError, AskInput, ModelOutput
from app.orchestration import Orchestrator
from app.storage import SQLiteStorage
from app.telemetry import Telemetry


class StudyService:
    def __init__(
        self,
        storage: SQLiteStorage,
        orchestrator: Orchestrator,
        telemetry: Telemetry,
    ):
        self.storage = storage
        self.orchestrator = orchestrator
        self.telemetry = telemetry

    def handle_question(
        self,
        data: AskInput,
        request_id: str | None = None,
    ) -> ModelOutput:

        request_id = request_id or str(
            uuid.uuid4()
        )

        self.telemetry.emit(
            request_id,
            "service",
            "started",
        )

        context = self.storage.load_documents(
            data.documents,
            request_id,
        )

        result = self.orchestrator.answer(
            data.question,
            data.documents,
            context,
            request_id,
        )

        self.telemetry.emit(
            request_id,
            "service",
            "finished",
        )

        return result

    def run_job(
        self,
        data: AskInput,
        request_id: str,
    ):
        self.storage.set_running(request_id)

        try:
            result = self.handle_question(
                data, request_id
            )

            self.storage.complete(
                request_id,
                result.model_dump(),
            )

        except AppError as err:
            self.storage.fail(
                request_id,
                {
                    "kind": err.kind,
                    "message": err.message,
                    "retryable": err.retryable,
                    "status_code": err.status_code,
                },
            )

        except Exception:
            self.storage.fail(
                request_id,
                {
                    "kind": "internal_error",
                    "message": (
                        "Внутрішня помилка сервера"
                    ),
                    "retryable": False,
                    "status_code": 503,
                },
            )