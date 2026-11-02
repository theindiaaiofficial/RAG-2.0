from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge.ingestion import KnowledgeIngestor  # noqa: E402
from app.memory.conversation_store import ConversationStore  # noqa: E402
from app.rag.embeddings import RemoteEmbeddingModel  # noqa: E402
from app.rag.vector_store import ChromaVectorStore  # noqa: E402


def main() -> int:
    settings = get_settings()
    store = ConversationStore(settings.sqlite_path)
    vector = ChromaVectorStore(settings.chroma_dir)
    vector.initialize()
    embedding_api_key = (
        settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None
    )
    embeddings = RemoteEmbeddingModel(
        settings.embedding_model,
        api_key=embedding_api_key,
        output_dimensionality=settings.embedding_output_dimensionality,
        batch_size=settings.embedding_batch_size,
    )
    embeddings.initialize()
    ingestor = KnowledgeIngestor(settings, embeddings, vector, store)
    result = ingestor.scan_and_ingest()
    print(json.dumps(result, indent=2))
    return 1 if result["failed_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
