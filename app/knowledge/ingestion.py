from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.knowledge.chunker import chunk_sections
from app.knowledge.parsers import SUPPORTED_EXTENSIONS, parse_file
from app.rag.embeddings import RemoteEmbeddingModel
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class KnowledgeIngestor:
    def __init__(self, settings: Settings, embeddings: RemoteEmbeddingModel, vector_store: ChromaVectorStore, store):
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.store = store
        self.last_errors: list[dict[str, str]] = []

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _files(self) -> list[Path]:
        root = self.settings.knowledge_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        found = []
        for path in root.rglob("*"):
            try:
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if path.name.lower() == "readme.md" and path.parent.resolve() == root:
                    continue
                resolved = path.resolve()
                if root not in resolved.parents:
                    logger.warning("Skipping path outside knowledge root")
                    continue
                found.append(resolved)
            except OSError:
                continue
        return sorted(found)

    def scan_and_ingest(self, full: bool = False) -> dict[str, Any]:
        self.last_errors = []
        root = self.settings.knowledge_dir.resolve()
        files = self._files()

        # A new/versioned vector collection can be empty while SQLite still has
        # file hashes from the previous embedding model. Rebuild rather than
        # incorrectly treating those old hashes as already indexed here.
        if not full and self.vector_store.count() == 0 and self.store.list_ingestion_paths():
            logger.info("Empty vector collection with prior ingestion records; forcing full reindex")
            full = True

        seen_paths: set[str] = set()
        processed = skipped = failed = chunks_added = 0
        now = datetime.now(timezone.utc).isoformat()

        if full:
            self.vector_store.delete_all()
            self.store.clear_ingestion_records()

        for path in files:
            relative = path.relative_to(root).as_posix()
            seen_paths.add(relative)
            try:
                file_hash = self._sha256(path)
                previous = self.store.get_ingestion_record(relative)
                if previous and previous["file_hash"] == file_hash and not full:
                    skipped += 1
                    continue
                # Remove any old version before parsing the changed file so corrupt edits cannot leave stale facts indexed.
                self.vector_store.delete_source(relative)
                sections = parse_file(path)
                chunks = chunk_sections(sections, self.settings.rag_chunk_size, self.settings.rag_chunk_overlap)
                if chunks:
                    records = []
                    for chunk in chunks:
                        chunk_id = hashlib.sha256(f"{relative}:{file_hash}:{chunk.index}".encode()).hexdigest()
                        records.append(
                            {
                                "id": chunk_id,
                                "text": chunk.text,
                                "metadata": {
                                    "source_filename": path.name,
                                    "relative_path": relative,
                                    "file_hash": file_hash,
                                    "chunk_id": chunk_id,
                                    "chunk_index": chunk.index,
                                    "page_number": chunk.page if chunk.page is not None else -1,
                                    "document_title": chunk.title or path.stem,
                                    "ingested_at": now,
                                },
                            }
                        )
                    vectors = self.embeddings.encode_documents([record["text"] for record in records])
                    self.vector_store.add_chunks(records, vectors)
                    chunks_added += len(records)
                self.store.upsert_ingestion_record(relative, file_hash, len(chunks), now)
                processed += 1
            except Exception as exc:
                failed += 1
                self.last_errors.append({"file": relative, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
                logger.exception("Knowledge file processing failed relative_path=%s", relative)

        # Remove stale sources that disappeared since the previous scan.
        known_paths = self.store.list_ingestion_paths()
        for relative in known_paths:
            if relative not in seen_paths:
                self.vector_store.delete_source(relative)
                self.store.delete_ingestion_record(relative)

        stats = {
            "discovered_files": len(files),
            "processed_files": processed,
            "unchanged_files": skipped,
            "failed_files": failed,
            "chunks_added": chunks_added,
            "document_count": len([p for p in self._files()]),
            "chunk_count": self.vector_store.count(),
            "errors": self.last_errors,
        }
        logger.info("Knowledge scan complete stats=%s", {k: v for k, v in stats.items() if k != "errors"})
        return stats

    def status(self) -> dict[str, Any]:
        files = self._files()
        indexed_paths = self.store.list_ingestion_paths()
        count = self.vector_store.count()
        empty = len(files) == 0
        return {
            "ready": not self.last_errors,
            "directory": str(self.settings.knowledge_dir),
            "empty": empty,
            "document_count": len(files),
            "chunk_count": count,
            "indexed_files": len(indexed_paths),
            "errors": self.last_errors,
            "message": (
                "No supported knowledge files found. Add .txt, .md, .pdf, .docx, .csv, or .json files."
                if empty else f"Knowledge base contains {len(files)} supported file(s) and {count} indexed chunk(s)."
            ),
        }
