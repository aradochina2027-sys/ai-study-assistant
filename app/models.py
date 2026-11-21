
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)


class AskInput(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
    )

    documents: list[str] = Field(
        default_factory=lambda: ["data/notes.md"],
        min_length=1,
        max_length=5,
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty"
            )

        return value


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "no_evidence"]
    answer: str
    sources: list[str]


class JobAccepted(BaseModel):
    request_id: str
    status: Literal["pending"] = "pending"
    status_url: str


class JobView(BaseModel):
    request_id: str

    status: Literal[
        "pending",
        "running",
        "completed",
        "cancelled",
    ]

    result: ModelOutput | None = None


class ErrorDetail(BaseModel):
    kind: str
    message: str
    retryable: bool
    request_id: str

    # ПР17 — помилки конкретних полів.
    fields: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail

    # Сумісність із поточним JavaScript,
    # який читає detail при HTTP 422.
    detail: list[dict] | None = None


class ProviderMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class ProviderRequest(BaseModel):
    request_id: str
    model: str
    messages: list[ProviderMessage]
    documents: list[str]
    timeout_s: float

    temperature: float = 0.0
    max_tokens: int = 450

    output_schema: dict = Field(
        default_factory=dict
    )

    prompt_version: str = "v1"


class ProviderResult(BaseModel):
    text: str
    model: str

    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0


class StorageRecord(BaseModel):
    request_id: str
    status: str

    result_json: str | None = None
    error_json: str | None = None


class AppError(Exception):
    def __init__(
        self,
        kind: str,
        message: str,
        status_code: int = 503,
        retryable: bool = False,
    ):
        super().__init__(message)

        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
