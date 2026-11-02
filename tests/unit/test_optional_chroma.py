from __future__ import annotations

import pytest

from app.rag.vector_store import ChromaVectorStore


def test_chroma_persistent_collection_survives_reopen(tmp_path):
    pytest.importorskip("chromadb")
    first = ChromaVectorStore(tmp_path / "chroma")
    first.initialize()
    first.add_chunks(
        [
            {
                "id": "a",
                "text": "persist me",
                "metadata": {"relative_path": "a.txt", "source_filename": "a.txt"},
            }
        ],
        [[1.0, 0.0]],
    )
    assert first.count() == 1
    second = ChromaVectorStore(tmp_path / "chroma")
    second.initialize()
    assert second.count() == 1
