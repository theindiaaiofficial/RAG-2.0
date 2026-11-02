from __future__ import annotations

from app.knowledge.ingestion import KnowledgeIngestor
from app.rag.retriever import RAGRetriever


class FakeEmbeddings:
    def encode_documents(self, texts):
        return [self.encode_query(text) for text in texts]

    def encode_query(self, text):
        # Deterministic test-only vectors; runtime uses the remote Gemini Embeddings API.
        vector = [0.0] * 32
        for char in text.casefold():
            vector[ord(char) % len(vector)] += 1.0
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]


class FakeVectorStore:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def delete_source(self, relative):
        self.records = {k: v for k, v in self.records.items() if v["metadata"]["relative_path"] != relative}

    def delete_all(self):
        self.records.clear()

    def add_chunks(self, records, embeddings):
        for record, vector in zip(records, embeddings):
            self.records[record["id"]] = {**record, "embedding": vector}

    def search(self, query_embedding, top_k):
        rows = []
        for key, item in self.records.items():
            a = item["embedding"]
            dot = sum(x * y for x, y in zip(a, query_embedding))
            rows.append(
                {
                    "id": key,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "distance": 1 - dot,
                    "similarity": dot,
                }
            )
        return sorted(rows, key=lambda x: x["distance"])[:top_k]

    def all_source_paths(self):
        return {v["metadata"]["relative_path"] for v in self.records.values()}


def test_ingestion_incremental_change_and_delete(tmp_path, settings):
    doc = settings.knowledge_dir / "services.txt"
    doc.write_text("North Star offers website design and mobile application development.", encoding="utf-8")
    store = FakeVectorStore()
    ingestor = KnowledgeIngestor(
        settings,
        FakeEmbeddings(),
        store,
        __import__("app.memory.conversation_store", fromlist=["ConversationStore"]).ConversationStore(
            settings.sqlite_path
        ),
    )
    first = ingestor.scan_and_ingest()
    assert first["processed_files"] == 1 and first["chunk_count"] > 0
    record_ids = set(store.records)
    second = ingestor.scan_and_ingest()
    assert second["unchanged_files"] == 1 and set(store.records) == record_ids
    doc.write_text("North Star offers cloud migration consulting.", encoding="utf-8")
    third = ingestor.scan_and_ingest()
    assert third["processed_files"] == 1
    assert all("cloud migration" in value["text"] for value in store.records.values())
    doc.unlink()
    fourth = ingestor.scan_and_ingest()
    assert fourth["chunk_count"] == 0 and not store.records


def test_ingestion_handles_corrupt_file_without_stopping(tmp_path, settings):
    (settings.knowledge_dir / "bad.pdf").write_text("bad pdf", encoding="utf-8")
    (settings.knowledge_dir / "good.txt").write_text("Business hours are Monday to Friday.", encoding="utf-8")
    store = FakeVectorStore()
    from app.memory.conversation_store import ConversationStore

    ingestor = KnowledgeIngestor(settings, FakeEmbeddings(), store, ConversationStore(settings.sqlite_path))
    stats = ingestor.scan_and_ingest()
    assert stats["failed_files"] == 1
    assert stats["processed_files"] == 1
    assert store.count() == 1


def test_retriever_top_k_duplicate_and_metadata(settings):
    store = FakeVectorStore()
    emb = FakeEmbeddings()
    records = []
    for i, text in enumerate(
        ["Company operates from Delhi", "Company operates from Delhi", "Support is available by email"]
    ):
        records.append(
            {
                "id": str(i),
                "text": text,
                "metadata": {
                    "source_filename": "facts.txt",
                    "relative_path": "facts.txt",
                    "page_number": -1,
                    "document_title": "Facts",
                },
            }
        )
    store.add_chunks(records, emb.encode_documents([r["text"] for r in records]))
    result = RAGRetriever(settings, emb, store).retrieve("Company operates from Delhi")
    assert len(result.chunks) <= settings.rag_top_k
    assert len({x["text"] for x in result.chunks}) == len(result.chunks)
    if result.chunks:
        assert result.chunks[0]["metadata"]["source_filename"] == "facts.txt"


def test_retriever_accepts_unicode_hindi_query(settings):
    store = FakeVectorStore()
    embeddings = FakeEmbeddings()
    text = "कंपनी सोमवार से शुक्रवार तक सहायता देती है।"
    record = {
        "id": "hindi",
        "text": text,
        "metadata": {
            "source_filename": "hindi.txt",
            "relative_path": "hindi.txt",
            "page_number": -1,
            "document_title": "Hindi",
        },
    }
    store.add_chunks([record], embeddings.encode_documents([text]))
    result = RAGRetriever(settings, embeddings, store).retrieve("कंपनी सहायता")
    assert result.chunks
    assert result.chunks[0]["metadata"]["source_filename"] == "hindi.txt"


def test_knowledge_scanner_skips_symlink_outside_root(tmp_path, settings):
    from app.memory.conversation_store import ConversationStore

    outside = tmp_path / "outside.txt"
    outside.write_text("Private external file", encoding="utf-8")
    link = settings.knowledge_dir / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        import pytest

        pytest.skip("Symlinks are unavailable on this platform")
    store = FakeVectorStore()
    ingestor = KnowledgeIngestor(settings, FakeEmbeddings(), store, ConversationStore(settings.sqlite_path))
    stats = ingestor.scan_and_ingest()
    assert stats["discovered_files"] == 0
    assert store.count() == 0



def test_empty_new_vector_collection_forces_reindex_with_existing_hashes(settings):
    doc = settings.knowledge_dir / "services.txt"
    doc.write_text("North Star offers website design.", encoding="utf-8")
    from app.memory.conversation_store import ConversationStore

    store = ConversationStore(settings.sqlite_path)
    vector = FakeVectorStore()
    ingestor = KnowledgeIngestor(settings, FakeEmbeddings(), vector, store)

    first = ingestor.scan_and_ingest()
    assert first["processed_files"] == 1
    assert store.list_ingestion_paths() == {"services.txt"}

    # Simulate switching to a fresh, versioned Chroma collection while keeping
    # the pre-existing SQLite ingestion-hash table.
    vector.delete_all()
    assert vector.count() == 0

    rebuilt = ingestor.scan_and_ingest()
    assert rebuilt["processed_files"] == 1
    assert rebuilt["unchanged_files"] == 0
    assert vector.count() > 0
