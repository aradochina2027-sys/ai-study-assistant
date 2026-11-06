from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskInput(BaseModel):
    question: str = Field(min_length=1, max_length=500)

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
            raise ValueError("Question cannot be empty")

        return value


class ModelOutput(BaseModel):
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
        "pending", "running", "completed", "cancelled"
    ]
    result: ModelOutput | None = None


class ErrorDetail(BaseModel):
    kind: str
    message: str
    retryable: bool
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ProviderMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class ProviderRequest(BaseModel):
    request_id: str
    model: str
    messages: list[ProviderMessage]
    documents: list[str]
    timeout_s: float


class ProviderResult(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


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