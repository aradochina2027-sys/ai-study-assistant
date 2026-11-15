import sqlite3

from datetime import datetime, timezone
from pathlib import Path

from app.models import AppError


class DailyBudget:
    def __init__(
        self,
        db_path: str,
        limit: int,
    ):
        self.path = Path(db_path)
        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.limit = limit

        with sqlite3.connect(self.path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS ai_budget (
                    day TEXT PRIMARY KEY,
                    calls INTEGER NOT NULL
                )
            """)

    def charge(self):
        today = datetime.now(
            timezone.utc
        ).date().isoformat()

        with sqlite3.connect(
            self.path,
            timeout=10,
            isolation_level=None,
        ) as db:

            db.execute("BEGIN IMMEDIATE")

            row = db.execute(
                "SELECT calls FROM ai_budget WHERE day=?",
                (today,),
            ).fetchone()

            used = row[0] if row else 0

            if used >= self.limit:
                db.rollback()

                raise AppError(
                    "budget",
                    "Вичерпано денний ліміт викликів AI",
                    429,
                )

            db.execute(
                """
                INSERT INTO ai_budget(day, calls)
                VALUES (?, 1)
                ON CONFLICT(day)
                DO UPDATE SET calls=calls+1
                """,
                (today,),
            )

            db.commit()