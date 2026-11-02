from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag.embeddings import RemoteEmbeddingModel


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append(
            {
                "model": model,
                "contents": list(contents),
                "config": config,
            }
        )
        vector = [3.0, 4.0] + [0.0] * 126
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=vector) for _ in contents]
        )


def make_embedding_model(batch_size=2):
    embedding = RemoteEmbeddingModel(
        "gemini-embedding-001",
        api_key="test-api-key",
        output_dimensionality=128,
        batch_size=batch_size,
    )
    models = FakeModels()
    embedding._client = SimpleNamespace(models=models)
    embedding._build_config = lambda task_type: {
        "task_type": task_type,
        "output_dimensionality": 128,
    }
    return embedding, models


def test_document_embeddings_are_batched_and_normalized():
    embedding, models = make_embedding_model(batch_size=2)

    vectors = embedding.encode_documents(["one", "two", "three"])

    assert len(vectors) == 3
    assert [len(call["contents"]) for call in models.calls] == [2, 1]
    assert all(call["config"]["task_type"] == "RETRIEVAL_DOCUMENT" for call in models.calls)
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)


def test_query_uses_retrieval_query_task_type():
    embedding, models = make_embedding_model()

    vector = embedding.encode_query("Where is the business located?")

    assert len(vector) == 128
    assert models.calls[0]["config"]["task_type"] == "RETRIEVAL_QUERY"
    assert models.calls[0]["contents"] == ["Where is the business located?"]


def test_empty_document_input_does_not_call_api():
    embedding, models = make_embedding_model()

    assert embedding.encode_documents([]) == []
    assert models.calls == []


def test_empty_query_is_rejected():
    embedding, _ = make_embedding_model()

    with pytest.raises(ValueError, match="cannot be empty"):
        embedding.encode_query("  ")


def test_unsupported_embedding_model_is_rejected():
    with pytest.raises(ValueError, match="gemini-embedding-001"):
        RemoteEmbeddingModel("gemini-embedding-2", api_key="test-api-key")


def test_initialize_requires_api_key():
    embedding = RemoteEmbeddingModel("gemini-embedding-001")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embedding.initialize()


def test_initialize_only_creates_lightweight_client():
    embedding = RemoteEmbeddingModel("gemini-embedding-001", api_key="test-api-key")
    marker = object()
    embedding._create_client = lambda: marker

    embedding.initialize()

    assert embedding.is_loaded is True
    assert embedding._client is marker


def test_legacy_local_model_name_is_migrated_in_settings(settings):
    from app.config import Settings

    migrated = Settings(
        router_model=settings.router_model,
        router_api_key=settings.router_api_key,
        answer_model=settings.answer_model,
        answer_api_key=settings.answer_api_key,
        embedding_api_key="test-gemini-key",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        knowledge_dir=settings.knowledge_dir,
        data_dir=settings.data_dir,
        chroma_dir=settings.chroma_dir,
        sqlite_path=settings.sqlite_path,
    )

    assert migrated.embedding_model == "gemini-embedding-001"
