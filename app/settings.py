
"""
Налаштування AI Study Assistant.

ПР15:
    AI-провайдер, модель, бюджет,
    промпти та mock.

ПР16:
    RAG та embeddings.

ПР18:
    AI_ENABLED — вимикач AI-функції.
"""

import os

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv


def read_bool_env(
    name: str,
    default: bool,
) -> bool:
    """
    Безпечне читання булевого налаштування.

    Підтримуються:
        true / false
        1 / 0
        yes / no
        on / off

    Помилкове значення не ігнорується.
    """

    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "true", "1", "yes", "on"
    }:
        return True

    if normalized in {
        "false", "0", "no", "off"
    }:
        return False

    raise ValueError(
        f"Некоректне значення {name}. "
        "Використовуйте true або false."
    )


@dataclass(frozen=True)
class Settings:
    # ==========================================
    # ПР15 — Основна модель
    # ==========================================

    llm_provider: str = "mock"
    llm_model: str = "mock-study-v1"
    llm_api_key: str = ""
    llm_base_url: str = (
        "http://localhost:11434/v1"
    )
    llm_timeout_s: float = 120.0

    # ==========================================
    # ПР18 — Вимикач AI
    # ==========================================

    ai_enabled: bool = True

    # ==========================================
    # Сховище та журналювання
    # ==========================================

    db_path: str = "data/study.db"
    log_file: str = "logs/app.jsonl"
    app_version: str = "pr15"

    # ==========================================
    # Mock-провайдер
    # ==========================================

    mock_delay_ms: int = 0
    mock_failure_rate: float = 0.0
    mock_failure_kind: str = "none"

    # ==========================================
    # Записи провайдера
    # ==========================================

    record_mode: str = "replay"
    recordings_dir: str = (
        "tests/recordings/study"
    )

    # ==========================================
    # Бюджет і навчальний контекст
    # ==========================================

    daily_call_limit: int = 40
    context_max_chars: int = 10000
    prompt_file: str = (
        "prompts/study/v1.md"
    )

    # ==========================================
    # ПР16 — RAG
    # ==========================================

    rag_enabled: bool = False

    rag_embedding_model: str = (
        "qwen3-embedding:0.6b"
    )

    rag_index_path: str = (
        ".rag_index/notes.json"
    )

    rag_top_k: int = 3
    rag_min_score: float = 0.25

    # ==========================================
    # Перевірка конфігурації
    # ==========================================

    def validate(self) -> None:
        allowed_providers = {
            "mock",
            "ollama",
            "openai_compat",
            "recorded",
        }

        if self.llm_provider not in allowed_providers:
            raise ValueError(
                "Invalid LLM_PROVIDER"
            )

        if (
            self.llm_provider == "openai_compat"
            and not self.llm_api_key
        ):
            raise ValueError(
                "LLM_API_KEY is required"
            )

        if not self.llm_model.strip():
            raise ValueError(
                "LLM_MODEL cannot be empty"
            )

        if (
            not isfinite(self.llm_timeout_s)
            or not 0 < self.llm_timeout_s <= 600
        ):
            raise ValueError(
                "Invalid LLM_TIMEOUT_S"
            )

        if not isinstance(self.ai_enabled, bool):
            raise ValueError(
                "Invalid AI_ENABLED"
            )

        if not 0 <= self.mock_failure_rate <= 1:
            raise ValueError(
                "Invalid MOCK_FAILURE_RATE"
            )

        if self.mock_failure_kind not in {
            "none",
            "timeout",
            "rate_limit",
            "unavailable",
            "invalid_json",
            "mid_stream",
        }:
            raise ValueError(
                "Invalid MOCK_FAILURE_KIND"
            )

        if self.record_mode not in {
            "replay",
            "record",
            "passthrough",
        }:
            raise ValueError(
                "Invalid RECORD_MODE"
            )

        if not 1 <= self.daily_call_limit <= 10000:
            raise ValueError(
                "Invalid DAILY_CALL_LIMIT"
            )

        if not 1000 <= self.context_max_chars <= 200000:
            raise ValueError(
                "Invalid CONTEXT_MAX_CHARS"
            )

        if not self.llm_base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "Invalid LLM_BASE_URL"
            )

        if not Path(
            self.prompt_file
        ).is_file():
            raise ValueError(
                "Prompt file not found: "
                + self.prompt_file
            )

        # ======================================
        # ПР16 — Перевірка RAG
        # ======================================

        if not isinstance(
            self.rag_enabled,
            bool,
        ):
            raise ValueError(
                "Invalid RAG_ENABLED"
            )

        if not self.rag_embedding_model.strip():
            raise ValueError(
                "RAG_EMBEDDING_MODEL cannot be empty"
            )

        if not self.rag_index_path.strip():
            raise ValueError(
                "RAG_INDEX_PATH cannot be empty"
            )

        if not 1 <= self.rag_top_k <= 10:
            raise ValueError(
                "RAG_TOP_K must be between 1 and 10"
            )

        if (
            not isfinite(self.rag_min_score)
            or not -1 <= self.rag_min_score <= 1
        ):
            raise ValueError(
                "RAG_MIN_SCORE must be between -1 and 1"
            )


def load_settings() -> Settings:
    """
    Читає змінні середовища та .env.

    Якщо AI_ENABLED=false,
    StudyService не запускає AI-сценарій.
    """

    load_dotenv()

    settings = Settings(
        # ======================================
        # Основна модель
        # ======================================

        llm_provider=os.getenv(
            "LLM_PROVIDER",
            "mock",
        ),

        llm_model=os.getenv(
            "LLM_MODEL",
            "mock-study-v1",
        ),

        llm_api_key=os.getenv(
            "LLM_API_KEY",
            "",
        ),

        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434/v1",
        ),

        llm_timeout_s=float(
            os.getenv(
                "LLM_TIMEOUT_S",
                "120",
            )
        ),

        # ======================================
        # ПР18 — Вимикач AI
        # ======================================

        ai_enabled=read_bool_env(
            "AI_ENABLED",
            True,
        ),

        # ======================================
        # Сховище та журнал
        # ======================================

        db_path=os.getenv(
            "DB_PATH",
            "data/study.db",
        ),

        log_file=os.getenv(
            "LOG_FILE",
            "logs/app.jsonl",
        ),

        app_version=os.getenv(
            "APP_VERSION",
            "pr15",
        ),

        # ======================================
        # Mock
        # ======================================

        mock_delay_ms=int(
            os.getenv(
                "MOCK_DELAY_MS",
                "0",
            )
        ),

        mock_failure_rate=float(
            os.getenv(
                "MOCK_FAILURE_RATE",
                "0",
            )
        ),

        mock_failure_kind=os.getenv(
            "MOCK_FAILURE_KIND",
            "none",
        ),

        # ======================================
        # Recorded provider
        # ======================================

        record_mode=os.getenv(
            "RECORD_MODE",
            "replay",
        ),

        recordings_dir=os.getenv(
            "RECORDINGS_DIR",
            "tests/recordings/study",
        ),

        # ======================================
        # Бюджет та контекст
        # ======================================

        daily_call_limit=int(
            os.getenv(
                "DAILY_CALL_LIMIT",
                "40",
            )
        ),

        context_max_chars=int(
            os.getenv(
                "CONTEXT_MAX_CHARS",
                "10000",
            )
        ),

        prompt_file=os.getenv(
            "PROMPT_FILE",
            "prompts/study/v1.md",
        ),

        # ======================================
        # ПР16 — RAG
        # ======================================

        rag_enabled=read_bool_env(
            "RAG_ENABLED",
            False,
        ),

        rag_embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "qwen3-embedding:0.6b",
        ),

        rag_index_path=os.getenv(
            "RAG_INDEX_PATH",
            ".rag_index/notes.json",
        ),

        rag_top_k=int(
            os.getenv(
                "RAG_TOP_K",
                "3",
            )
        ),

        rag_min_score=float(
            os.getenv(
                "RAG_MIN_SCORE",
                "0.25",
            )
        ),
    )

    settings.validate()

    return settings
