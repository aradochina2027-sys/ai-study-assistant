from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    llm_model: str = "mock-study-v1"
    llm_api_key: str = ""
    llm_timeout_s: float = 15.0

    db_path: str = "data/study.db"
    log_file: str = "logs/app.jsonl"
    app_version: str = "pr14"

    mock_delay_ms: int = 0
    mock_failure_rate: float = 0.0
    mock_failure_kind: str = "none"

    def validate(self):
        providers = {"mock", "ollama", "openai_compat"}

        if self.llm_provider not in providers:
            raise ValueError("Invalid LLM_PROVIDER")

        if (
            self.llm_provider == "openai_compat"
            and not self.llm_api_key
        ):
            raise ValueError("LLM_API_KEY is required")

        if not self.llm_model.strip():
            raise ValueError("LLM_MODEL cannot be empty")

        if not 0 < self.llm_timeout_s <= 300:
            raise ValueError("Invalid LLM_TIMEOUT_S")

        if not 0 <= self.mock_failure_rate <= 1:
            raise ValueError("Invalid MOCK_FAILURE_RATE")

        if not 0 <= self.mock_delay_ms <= 120000:
            raise ValueError("Invalid MOCK_DELAY_MS")

        failures = {
            "none", "timeout", "rate_limit",
            "unavailable", "invalid_json", "mid_stream"
        }

        if self.mock_failure_kind not in failures:
            raise ValueError("Invalid MOCK_FAILURE_KIND")


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        llm_model=os.getenv("LLM_MODEL", "mock-study-v1"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "15")),
        db_path=os.getenv("DB_PATH", "data/study.db"),
        log_file=os.getenv("LOG_FILE", "logs/app.jsonl"),
        app_version=os.getenv("APP_VERSION", "pr14"),
        mock_delay_ms=int(os.getenv("MOCK_DELAY_MS", "0")),
        mock_failure_rate=float(
            os.getenv("MOCK_FAILURE_RATE", "0")
        ),
        mock_failure_kind=os.getenv(
            "MOCK_FAILURE_KIND", "none"
        ),
    )

    settings.validate()
    return settings