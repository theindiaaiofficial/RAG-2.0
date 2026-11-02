from __future__ import annotations

from pathlib import Path
from typing import Any


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "business_knowledge_gemini001_768"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def initialize(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt") from exc
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Business knowledge chunks embedded with Gemini Embedding 001"},
        )

    @property
    def collection(self):
        if self._collection is None:
            raise RuntimeError("Vector store is not initialized")
        return self._collection

    def count(self) -> int:
        return int(self.collection.count())

    def delete_source(self, relative_path: str) -> None:
        self.collection.delete(where={"relative_path": relative_path})

    def delete_all(self) -> None:
        if self._client is None:
            self.initialize()
        assert self._client is not None
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, records: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not records:
            return
        if len(records) != len(embeddings):
            raise ValueError("Each record must have exactly one embedding")
        self.collection.add(
            ids=[record["id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=embeddings,
            metadatas=[record["metadata"] for record in records],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        output = []
        for doc_id, text, metadata, distance in zip(ids, docs, metas, distances):
            output.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": float(distance),
                    "similarity": max(0.0, min(1.0, 1.0 - float(distance))),
                }
            )
        return output

    def all_source_paths(self) -> set[str]:
        if self.count() == 0:
            return set()
        result = self.collection.get(include=["metadatas"])
        return {
            str(metadata.get("relative_path"))
            for metadata in (result.get("metadatas") or [])
            if metadata and metadata.get("relative_path")
        }

    def close(self) -> None:
        """Release the Chroma client and its persistent file handles."""
        client = self._client
        self._collection = None
        self._client = None

        if client is not None:
            # Chroma PersistentClient versions differ: some expose close(),
            # while others require stopping their underlying System.
            close = getattr(client, "close", None)
            if callable(close):
                close()
            else:
                system = getattr(client, "_system", None)
                stop = getattr(system, "stop", None)
                if callable(stop):
                    stop()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
