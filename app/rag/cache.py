
"""ПР18: потокобезпечний TTL-кеш embedding-векторів у пам'яті процесу.

Ключ зберігається лише як SHA-256, без тексту документа.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

from collections import OrderedDict
from threading import RLock
from typing import Callable


class EmbeddingCache:
    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_entries: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds має бути додатним скінченним числом"
            )

        if type(max_entries) is not int or max_entries < 1:
            raise ValueError(
                "max_entries має бути додатним цілим числом"
            )

        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max_entries
        self._clock = clock

        self._items: OrderedDict[
            str,
            tuple[float, list[float], str],
        ] = OrderedDict()

        self._lock = RLock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _key(
        text: str,
        model: str,
        owner_id: str,
        revision: str,
        base_url: str,
    ) -> str:
        if not owner_id or not owner_id.strip():
            raise ValueError(
                "Для кешу потрібен owner_id"
            )

        data = json.dumps(
            [
                "embedding-v1",
                owner_id,
                model,
                revision,
                base_url.rstrip("/"),
                text,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            data.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        text: str,
        *,
        model: str,
        owner_id: str,
        revision: str = "v1",
        base_url: str = "http://localhost:11434",
    ) -> list[float] | None:
        key = self._key(
            text,
            model,
            owner_id,
            revision,
            base_url,
        )

        now = self._clock()

        with self._lock:
            found = self._items.get(key)

            if found is None:
                self._misses += 1
                return None

            expiry, vector, _ = found

            if now >= expiry:
                del self._items[key]
                self._misses += 1
                return None

            self._hits += 1

            self._items.move_to_end(key)

            return vector.copy()

    def put(
        self,
        text: str,
        vector: list[float],
        *,
        model: str,
        owner_id: str,
        revision: str = "v1",
        base_url: str = "http://localhost:11434",
    ) -> None:
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError(
                "Кешувати можна лише коректний embedding-вектор"
            )

        key = self._key(
            text,
            model,
            owner_id,
            revision,
            base_url,
        )

        values = [
            float(value)
            for value in vector
        ]

        with self._lock:
            self._items[key] = (
                self._clock() + self.ttl_seconds,
                values,
                owner_id,
            )

            self._items.move_to_end(key)

            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
                self._evictions += 1

    def invalidate_owner(
        self,
        owner_id: str,
    ) -> int:
        """Видаляє вектори власника після зміни прав доступу."""

        with self._lock:
            keys = [
                key
                for key, item in self._items.items()
                if item[2] == owner_id
            ]

            for key in keys:
                del self._items[key]

            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self._hits + self._misses

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (
                    round(self._hits / total, 4)
                    if total
                    else 0.0
                ),
                "entries": len(self._items),
                "evictions": self._evictions,
            }
