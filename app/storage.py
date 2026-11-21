
import json
import sqlite3
from pathlib import Path

from app.models import AppError, StorageRecord
from app.telemetry import Telemetry


class SQLiteStorage:
    """
    SQLite-сховище AI Study Assistant.

    ПР14–ПР16:
    - збереження запитів;
    - стани pending/running/completed/failed/cancelled;
    - результати та помилки;
    - завантаження документів.

    ПР17:
    - захист від дублювання HTTP-запитів;
    - постійне збереження Idempotency-Key;
    - атомарна реєстрація нових завдань.
    """

    def __init__(
        self,
        filename: str,
        telemetry: Telemetry,
    ):
        self.path = Path(filename)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.telemetry = telemetry

        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:

            db.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT
                )
                """
            )

            # ПР17 — окрема таблиця ключів.
            # Існуючі записи requests зберігаються.

            db.execute(
                """
                CREATE TABLE IF NOT EXISTS request_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL
                )
                """
            )

    # ==========================================
    # ПР14 — Створення звичайного запиту
    # ==========================================

    def create(self, request_id: str):
        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            db.execute(
                """
                INSERT INTO requests (
                    request_id,
                    status
                )
                VALUES (?, 'pending')
                """,
                (request_id,),
            )

        self.telemetry.emit(
            request_id,
            "storage",
            "created",
        )

    # ==========================================
    # ПР17 — Ідемпотентне створення запиту
    # ==========================================

    def create_idempotent(
        self,
        request_id: str,
        idempotency_key: str,
        payload_hash: str,
    ) -> tuple[str, bool]:
        """
        Повертає:
            (request_id, created)

        created=True:
            створено новий запит.

        created=False:
            ключ уже існує, повертаємо
            попередній request_id.

        Якщо ключ використаний з іншими
        даними — повертаємо 409 Conflict.

        BEGIN IMMEDIATE захищає реєстрацію
        від гонитви двох одночасних запитів.
        """

        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:

            db.execute("BEGIN IMMEDIATE")

            existing = db.execute(
                """
                SELECT
                    request_id,
                    payload_hash
                FROM request_keys
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if existing is not None:
                previous_request_id = existing[0]
                previous_payload_hash = existing[1]

                if previous_payload_hash != payload_hash:
                    raise AppError(
                        kind="idempotency_conflict",
                        message=(
                            "Цей Idempotency-Key вже "
                            "використано з іншими даними. "
                            "Створіть новий ключ."
                        ),
                        status_code=409,
                        retryable=False,
                    )

                return previous_request_id, False

            db.execute(
                """
                INSERT INTO requests (
                    request_id,
                    status
                )
                VALUES (?, 'pending')
                """,
                (request_id,),
            )

            db.execute(
                """
                INSERT INTO request_keys (
                    idempotency_key,
                    request_id,
                    payload_hash
                )
                VALUES (?, ?, ?)
                """,
                (
                    idempotency_key,
                    request_id,
                    payload_hash,
                ),
            )

        self.telemetry.emit(
            request_id,
            "storage",
            "created_idempotent",
        )

        return request_id, True

    # ==========================================
    # Запит переходить у running
    # ==========================================

    def set_running(self, request_id: str):
        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            cursor = db.execute(
                """
                UPDATE requests
                SET status = 'running'
                WHERE request_id = ?
                  AND status = 'pending'
                """,
                (request_id,),
            )

        if cursor.rowcount > 0:
            self.telemetry.emit(
                request_id,
                "storage",
                "running",
            )

    # ==========================================
    # Збереження результату
    # ==========================================

    def complete(
        self,
        request_id: str,
        result: dict,
    ):
        result_json = json.dumps(
            result,
            ensure_ascii=False,
        )

        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            cursor = db.execute(
                """
                UPDATE requests
                SET
                    status = 'completed',
                    result_json = ?
                WHERE request_id = ?
                  AND status = 'running'
                """,
                (
                    result_json,
                    request_id,
                ),
            )

        if cursor.rowcount > 0:
            self.telemetry.emit(
                request_id,
                "storage",
                "completed",
            )

    # ==========================================
    # Збереження помилки
    # ==========================================

    def fail(
        self,
        request_id: str,
        error: dict,
    ):
        error_json = json.dumps(
            error,
            ensure_ascii=False,
        )

        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            cursor = db.execute(
                """
                UPDATE requests
                SET
                    status = 'failed',
                    error_json = ?
                WHERE request_id = ?
                  AND status = 'running'
                """,
                (
                    error_json,
                    request_id,
                ),
            )

        if cursor.rowcount > 0:
            self.telemetry.emit(
                request_id,
                "storage",
                "failed",
            )

    # ==========================================
    # Скасування запиту
    # ==========================================

    def cancel(self, request_id: str) -> bool:
        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            cursor = db.execute(
                """
                UPDATE requests
                SET status = 'cancelled'
                WHERE request_id = ?
                  AND status IN ('pending', 'running')
                """,
                (request_id,),
            )

        cancelled = cursor.rowcount > 0

        if cancelled:
            self.telemetry.emit(
                request_id,
                "storage",
                "cancel_requested",
            )

        return cancelled

    # ==========================================
    # Читання запиту
    # ==========================================

    def get(
        self,
        request_id: str,
    ) -> StorageRecord | None:

        with sqlite3.connect(
            self.path,
            timeout=30,
        ) as db:
            db.row_factory = sqlite3.Row

            row = db.execute(
                """
                SELECT
                    request_id,
                    status,
                    result_json,
                    error_json
                FROM requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        self.telemetry.emit(
            request_id,
            "storage",
            "read",
        )

        if row is None:
            return None

        return StorageRecord(
            **dict(row)
        )

    # ==========================================
    # Завантаження навчальних документів
    # ==========================================

    def load_documents(
        self,
        names: list[str],
        request_id: str,
    ) -> str:

        base = Path("data").resolve()
        chunks: list[str] = []

        for name in names:
            filename = Path(name).resolve()

            if (
                not filename.is_relative_to(base)
                or filename.suffix.lower()
                not in {".md", ".txt"}
            ):
                raise AppError(
                    "invalid_document",
                    (
                        "Дозволені лише .md/.txt "
                        "файли з папки data"
                    ),
                    400,
                )

            if not filename.is_file():
                raise AppError(
                    "document_missing",
                    f"Документ {name} не знайдено",
                    404,
                )

            content = filename.read_text(
                encoding="utf-8",
            )

            if len(content) > 20000:
                raise AppError(
                    "document_too_large",
                    (
                        "Документ перевищує "
                        "20000 символів"
                    ),
                    400,
                )

            chunks.append(
                f"<<<DOC:{name}>>>\n"
                f"{content}\n"
                f"<<<END_DOC>>>"
            )

        self.telemetry.emit(
            request_id,
            "storage",
            "documents_loaded",
        )

        return "\n\n".join(chunks)

    # ==========================================
    # Перевірка готовності SQLite
    # ==========================================

    def ready(self) -> bool:
        try:
            with sqlite3.connect(
                self.path,
                timeout=30,
            ) as db:
                db.execute("SELECT 1")

            return True

        except sqlite3.Error:
            return False
