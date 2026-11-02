from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS ingestion_files (
                    relative_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                );
                """
            )

    def create_session(self, title: str = "New conversation", session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations(session_id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (session_id, title[:120], now, now),
            )
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM conversations WHERE session_id=?", (session_id,)).fetchone()
        return row is not None

    def save_message(
        self, session_id: str, role: str, content: str, citations: list[dict[str, Any]] | None = None
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be user or assistant")
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id,role,content,citations_json,created_at) VALUES(?,?,?,?,?)",
                (session_id, role, content, json.dumps(citations or [], ensure_ascii=False), now),
            )
            if role == "user":
                first = conn.execute(
                    "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id LIMIT 1",
                    (session_id,),
                ).fetchone()
                title = (first["content"][:72] if first else "New conversation") or "New conversation"
                conn.execute(
                    "UPDATE conversations SET title=?,updated_at=? WHERE session_id=?",
                    (title, now, session_id),
                )
            else:
                conn.execute("UPDATE conversations SET updated_at=? WHERE session_id=?", (now, session_id))

    def get_history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def get_conversation(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conversation = conn.execute(
                "SELECT session_id,title FROM conversations WHERE session_id=?", (session_id,)
            ).fetchone()
            if not conversation:
                return None
            rows = conn.execute(
                "SELECT role,content,citations_json,created_at FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        messages = []
        for row in rows:
            try:
                citations = json.loads(row["citations_json"])
            except json.JSONDecodeError:
                citations = []
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "citations": citations,
                    "created_at": row["created_at"],
                }
            )
        return {
            "session_id": conversation["session_id"],
            "title": conversation["title"],
            "messages": messages,
        }

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.session_id,c.title,c.created_at,c.updated_at,
                          COUNT(m.id) AS message_count
                   FROM conversations c LEFT JOIN messages m ON m.session_id=c.session_id
                   GROUP BY c.session_id ORDER BY c.updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ingestion_record(self, relative_path: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_files WHERE relative_path=?", (relative_path,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_ingestion_record(
        self, relative_path: str, file_hash: str, chunk_count: int, ingested_at: str
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO ingestion_files(relative_path,file_hash,chunk_count,ingested_at)
                   VALUES(?,?,?,?) ON CONFLICT(relative_path) DO UPDATE SET
                   file_hash=excluded.file_hash,chunk_count=excluded.chunk_count,ingested_at=excluded.ingested_at""",
                (relative_path, file_hash, chunk_count, ingested_at),
            )

    def list_ingestion_paths(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT relative_path FROM ingestion_files").fetchall()
        return {row["relative_path"] for row in rows}

    def delete_ingestion_record(self, relative_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM ingestion_files WHERE relative_path=?", (relative_path,))

    def clear_ingestion_records(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM ingestion_files")
