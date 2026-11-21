
"""ПР18: TTL, ізоляція, обхід та відсутність кешування помилок."""

import json

import httpx
import pytest

from app.rag.cache import EmbeddingCache
from app.rag.embedder import (
    EmbeddingError,
    embed_texts,
)


def provider(
    calls,
    *,
    bad_first=False,
):
    def handler(request):
        payload = json.loads(
            request.content
        )

        calls.append(payload)

        if bad_first and len(calls) == 1:
            return httpx.Response(503)

        values = [
            [float(len(text)), 1.0]
            for text in payload["input"]
        ]

        return httpx.Response(
            200,
            json={
                "embeddings": values
            },
        )

    return httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )


@pytest.fixture(autouse=True)
def enable_cache(monkeypatch):
    monkeypatch.setenv(
        "STUDYAI_EMBED_CACHE_ENABLED",
        "1",
    )

    monkeypatch.setenv(
        "STUDYAI_EMBED_CACHE_MODEL_REVISION",
        "v1",
    )


def test_second_call_hits_cache():
    calls = []
    cache = EmbeddingCache()

    with provider(calls) as client:
        first = embed_texts(
            ["Python"],
            client=client,
            cache=cache,
            cache_owner="user-a",
        )

        second = embed_texts(
            ["Python"],
            client=client,
            cache=cache,
            cache_owner="user-a",
        )

    assert first == second
    assert len(calls) == 1

    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1

    assert "Python" not in str(
        cache._items.keys()
    )


def test_partial_hit_preserves_order():
    calls = []
    cache = EmbeddingCache()

    with provider(calls) as client:
        first = embed_texts(
            ["A", "BBBB"],
            client=client,
            cache=cache,
            cache_owner="local",
        )

        second = embed_texts(
            ["BBBB", "CCC", "BBBB"],
            client=client,
            cache=cache,
            cache_owner="local",
        )

    assert first == [
        [1.0, 1.0],
        [4.0, 1.0],
    ]

    assert second == [
        [4.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
    ]

    assert len(calls) == 2

    assert calls[1]["input"] == [
        "CCC"
    ]


def test_different_owner_model_revision_or_text_misses(
    monkeypatch,
):
    calls = []
    cache = EmbeddingCache()

    with provider(calls) as client:
        combinations = [
            ("owner-a", "model-a", "v1", "text"),
            ("owner-b", "model-a", "v1", "text"),
            ("owner-a", "model-b", "v1", "text"),
            ("owner-a", "model-a", "v2", "text"),
            ("owner-a", "model-a", "v1", "text changed"),
        ]

        for owner, model, revision, text in combinations:
            monkeypatch.setenv(
                "STUDYAI_EMBED_CACHE_MODEL_REVISION",
                revision,
            )

            embed_texts(
                [text],
                model=model,
                client=client,
                cache=cache,
                cache_owner=owner,
            )

    assert len(calls) == 5


def test_ttl_expires_and_invalidation():
    clock = [10.0]

    cache = EmbeddingCache(
        ttl_seconds=5,
        clock=lambda: clock[0],
    )

    args = {
        "model": "a",
        "owner_id": "user-a",
    }

    cache.put(
        "text",
        [0.1, 0.2],
        **args,
    )

    assert cache.get(
        "text",
        **args,
    ) == [0.1, 0.2]

    clock[0] = 16.0

    assert cache.get(
        "text",
        **args,
    ) is None

    cache.put(
        "text",
        [0.1, 0.2],
        **args,
    )

    assert cache.invalidate_owner(
        "user-a"
    ) == 1

    assert cache.get(
        "text",
        **args,
    ) is None


def test_bypass_always_calls_provider():
    calls = []
    cache = EmbeddingCache()

    with provider(calls) as client:
        args = {
            "client": client,
            "cache": cache,
            "cache_owner": "local",
        }

        embed_texts(
            ["Git"],
            **args,
        )

        embed_texts(
            ["Git"],
            bypass_cache=True,
            **args,
        )

        embed_texts(
            ["Git"],
            **args,
        )

    assert len(calls) == 2
    assert cache.stats()["hits"] == 1


def test_errors_are_not_cached():
    calls = []
    cache = EmbeddingCache()

    with provider(
        calls,
        bad_first=True,
    ) as client:
        with pytest.raises(
            EmbeddingError
        ):
            embed_texts(
                ["SQL"],
                client=client,
                cache=cache,
                cache_owner="local",
            )

        assert cache.stats()["entries"] == 0

        embed_texts(
            ["SQL"],
            client=client,
            cache=cache,
            cache_owner="local",
        )

    assert len(calls) == 2


def test_cache_requires_owner_and_can_be_disabled(
    monkeypatch,
):
    calls = []
    cache = EmbeddingCache()

    monkeypatch.delenv(
        "STUDYAI_EMBED_CACHE_OWNER",
        raising=False,
    )

    with provider(calls) as client:
        with pytest.raises(ValueError):
            embed_texts(
                ["No owner"],
                client=client,
                cache=cache,
            )

        monkeypatch.setenv(
            "STUDYAI_EMBED_CACHE_ENABLED",
            "0",
        )

        embed_texts(
            ["same"],
            client=client,
            cache=cache,
        )

        embed_texts(
            ["same"],
            client=client,
            cache=cache,
        )

    assert len(calls) == 2
    assert cache.stats()["entries"] == 0


def test_cache_does_not_expose_mutable_vector():
    cache = EmbeddingCache()

    kwargs = {
        "model": "model",
        "owner_id": "owner",
    }

    cache.put(
        "text",
        [1, 2],
        **kwargs,
    )

    vector = cache.get(
        "text",
        **kwargs,
    )

    vector.append(3.0)

    assert cache.get(
        "text",
        **kwargs,
    ) == [1.0, 2.0]


def test_cache_limit_evicts_oldest():
    cache = EmbeddingCache(
        max_entries=1
    )

    kwargs = {
        "model": "model",
        "owner_id": "owner",
    }

    cache.put(
        "A",
        [1.0],
        **kwargs,
    )

    cache.put(
        "B",
        [2.0],
        **kwargs,
    )

    assert cache.get(
        "A",
        **kwargs,
    ) is None

    assert cache.stats()["evictions"] == 1
