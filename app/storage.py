import json
import sqlite3
from pathlib import Path

from app.models import AppError, StorageRecord
from app.telemetry import Telemetry


class SQLiteStorage:
    def __init__(
        self,
        filename: str,
        telemetry: Telemetry,
    ):
        self.path = Path(filename)
        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.telemetry = telemetry

        with sqlite3.connect(self.path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT
                )
            """)

    def create(self, request_id: str):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO requests "
                "(request_id,status) VALUES (?,?)",
                (request_id, "pending"),
            )

        self.telemetry.emit(
            request_id, "storage", "created"
        )

    def set_running(self, request_id: str):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE requests SET status='running' "
                "WHERE request_id=? AND status='pending'",
                (request_id,),
            )

        self.telemetry.emit(
            request_id, "storage", "running"
        )

    def complete(self, request_id: str, result: dict):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE requests "
                "SET status='completed', result_json=? "
                "WHERE request_id=? AND status='running'",
                (
                    json.dumps(
                        result, ensure_ascii=False
                    ),
                    request_id,
                ),
            )

        self.telemetry.emit(
            request_id, "storage", "completed"
        )

    def fail(self, request_id: str, error: dict):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE requests "
                "SET status='failed', error_json=? "
                "WHERE request_id=? AND status='running'",
                (
                    json.dumps(
                        error, ensure_ascii=False
                    ),
                    request_id,
                ),
            )

        self.telemetry.emit(
            request_id, "storage", "failed"
        )

    def cancel(self, request_id: str) -> bool:
        with sqlite3.connect(self.path) as db:
            cursor = db.execute(
                "UPDATE requests SET status='cancelled' "
                "WHERE request_id=? "
                "AND status IN ('pending','running')",
                (request_id,),
            )

        self.telemetry.emit(
            request_id, "storage", "cancel_requested"
        )

        return cursor.rowcount > 0

    def get(
        self, request_id: str
    ) -> StorageRecord | None:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM requests "
                "WHERE request_id=?",
                (request_id,),
            ).fetchone()

        self.telemetry.emit(
            request_id, "storage", "read"
        )

        return (
            StorageRecord(**dict(row))
            if row else None
        )

    def load_documents(
        self,
        names: list[str],
        request_id: str,
    ) -> str:
        base = Path("data").resolve()
        chunks = []

        for name in names:
            filename = Path(name).resolve()

            if (
                not filename.is_relative_to(base)
                or filename.suffix.lower()
                not in {".md", ".txt"}
            ):
                raise AppError(
                    "invalid_document",
                    "Дозволені лише .md/.txt "
                    "файли з папки data",
                    400,
                )

            if not filename.is_file():
                raise AppError(
                    "document_missing",
                    f"Документ {name} не знайдено",
                    404,
                )

            content = filename.read_text(
                encoding="utf-8"
            )

            if len(content) > 20000:
                raise AppError(
                    "document_too_large",
                    "Документ перевищує "
                    "20000 символів",
                    400,
                )

            chunks.append(content)

        self.telemetry.emit(
            request_id,
            "storage",
            "documents_loaded",
        )

        return "\n\n".join(chunks)

    def ready(self) -> bool:
        try:
            with sqlite3.connect(self.path) as db:
                db.execute("SELECT 1")

            return True

        except sqlite3.Error:
            return False