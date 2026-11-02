from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    return Settings(
        router_model="openai/gpt-4o-mini",
        router_api_key="test-router-key",
        answer_model="openai/gpt-4.1",
        answer_api_key="test-answer-key",
        embedding_api_key="test-gemini-key",
        knowledge_dir=knowledge,
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        sqlite_path=tmp_path / "data" / "assistant.db",
        rag_chunk_size=240,
        rag_chunk_overlap=40,
        rag_min_similarity=0.05,
    )
