from __future__ import annotations

from dataclasses import dataclass
import time

from app.config import Settings
from app.rag.embeddings import RemoteEmbeddingModel
from app.rag.vector_store import ChromaVectorStore


@dataclass
class RetrievalResult:
    chunks: list[dict]
    context: str
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0


class RAGRetriever:
    def __init__(self, settings: Settings, embeddings: RemoteEmbeddingModel, vector_store: ChromaVectorStore):
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(self, query: str) -> RetrievalResult:
        if self.vector_store.count() == 0:
            return RetrievalResult(chunks=[], context="")
        start = time.perf_counter()
        vector = self.embeddings.encode_query(query)
        embedding_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        hits = self.vector_store.search(vector, self.settings.rag_top_k)
        vector_search_ms = (time.perf_counter() - start) * 1000
        # Chroma's cosine distance is converted to similarity. Filter weak matches and duplicate text.
        selected = []
        seen_text: set[str] = set()
        for hit in hits:
            normalized = " ".join(hit["text"].casefold().split())
            if normalized in seen_text or hit["similarity"] < self.settings.rag_min_similarity:
                continue
            seen_text.add(normalized)
            selected.append(hit)
        context_parts = []
        used = 0
        for index, hit in enumerate(selected, start=1):
            meta = hit["metadata"]
            page = meta.get("page_number", -1)
            page_label = f", page {page}" if isinstance(page, int) and page > 0 else ""
            block = (
                f"[S{index}] source={meta.get('source_filename', 'unknown')}{page_label}; "
                f"title={meta.get('document_title', '')}; similarity={hit['similarity']:.3f}\n"
                f"{hit['text']}"
            )
            if used + len(block) > self.settings.rag_max_context_chars:
                break
            context_parts.append(block)
            hit["source_id"] = f"S{index}"
            used += len(block)
        # Keep only chunks actually included in the prompt.
        included_count = len(context_parts)
        selected = selected[:included_count]
        return RetrievalResult(
            chunks=selected,
            context="\n\n---\n\n".join(context_parts),
            embedding_ms=embedding_ms,
            vector_search_ms=vector_search_ms,
        )
