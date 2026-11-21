
"""ПР18: порівняння кількості embedding-звернень без Ollama."""

import json
import os

import httpx

from app.rag.cache import EmbeddingCache
from app.rag.embedder import embed_texts


def main() -> None:
    os.environ[
        "STUDYAI_EMBED_CACHE_ENABLED"
    ] = "1"

    os.environ[
        "STUDYAI_EMBED_CACHE_MODEL_REVISION"
    ] = "v1"

    owner = "local-pr18-demo"
    question = "Що робить git push?"

    calls = {
        "count": 0
    }

    def respond(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(
            request.content
        )

        calls["count"] += 1

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [float(len(text)), 1.0]
                    for text in payload["input"]
                ]
            },
        )

    transport = httpx.MockTransport(
        respond
    )

    with httpx.Client(
        transport=transport
    ) as client:
        cache = EmbeddingCache(
            ttl_seconds=3600
        )

        # П'ять однакових звернень
        # із примусовим обходом кешу.
        for _ in range(5):
            embed_texts(
                [question],
                client=client,
                cache=cache,
                cache_owner=owner,
                bypass_cache=True,
            )

        without_cache = calls["count"]

        calls["count"] = 0

        # П'ять таких самих звернень
        # зі звичайним кешуванням.
        for _ in range(5):
            embed_texts(
                [question],
                client=client,
                cache=cache,
                cache_owner=owner,
            )

        with_cache = calls["count"]

    saved = without_cache - with_cache

    saving_percent = (
        100 * saved / without_cache
        if without_cache
        else 0
    )

    print(
        f"Без кешу: {without_cache} HTTP-викликів"
    )

    print(
        f"З кешем: {with_cache} HTTP-викликів"
    )

    print(
        f"Зекономлено: {saved} "
        f"({saving_percent:.0f}%)"
    )

    print(
        f"Статистика кешу: {cache.stats()}"
    )

    print(
        "Це тест із підставним провайдером, "
        "а не вимірювання реальної Ollama."
    )


if __name__ == "__main__":
    main()
