import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class Telemetry:
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.filename.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.lock = Lock()

    def emit(
        self,
        request_id: str,
        layer: str,
        event: str,
    ):
        entry = {
            "ts": datetime.now(
                timezone.utc
            ).isoformat(),
            "request_id": request_id,
            "layer": layer,
            "event": event,
        }

        with self.lock:
            with self.filename.open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps(
                        entry, ensure_ascii=False
                    ) + "\n"
                )