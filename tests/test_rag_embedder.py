
import httpx
import pytest

from app.rag.embedder import (
    EmbeddingError,
    embed_texts,
)


def make_client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler)
    )


def test_embedding_success():
    def handler(request):
        assert request.url.path == "/api/embed"

        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            },
        )

    with make_client(handler) as client:
        vectors = embed_texts(
            ["Перший текст", "Другий текст"],
            client=client,
        )

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert vectors[0] == [0.1, 0.2, 0.3]


def test_empty_input():
    vectors = embed_texts([])

    assert vectors == []


def test_invalid_input():
    with pytest.raises(ValueError):
        embed_texts([""])

    with pytest.raises(ValueError):
        embed_texts(["   "])


def test_provider_unavailable():
    def handler(request):
        return httpx.Response(
            503,
            json={"error": "unavailable"},
        )

    with make_client(handler) as client:
        with pytest.raises(EmbeddingError):
            embed_texts(
                ["Тест"],
                client=client,
            )


def test_invalid_vector_count():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                ]
            },
        )

    with make_client(handler) as client:
        with pytest.raises(EmbeddingError):
            embed_texts(
                ["Перший", "Другий"],
                client=client,
            )


def test_inconsistent_dimensions():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ]
            },
        )

    with make_client(handler) as client:
        with pytest.raises(EmbeddingError):
            embed_texts(
                ["Перший", "Другий"],
                client=client,
            )
