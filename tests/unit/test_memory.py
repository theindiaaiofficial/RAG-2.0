from __future__ import annotations

from app.memory.conversation_store import ConversationStore


def test_conversation_persistence_and_order(tmp_path):
    path = tmp_path / "test.db"
    store = ConversationStore(path)
    session = store.create_session()
    store.save_message(session, "user", "Hello")
    store.save_message(session, "assistant", "Hi there")
    reopened = ConversationStore(path)
    history = reopened.get_history(session, 10)
    assert history == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
    assert reopened.get_conversation(session)["messages"][1]["content"] == "Hi there"


def test_session_isolation_and_history_limit(tmp_path):
    store = ConversationStore(tmp_path / "test.db")
    first = store.create_session()
    second = store.create_session()
    for index in range(5):
        store.save_message(first, "user", f"first {index}")
    store.save_message(second, "user", "only second")
    assert len(store.get_history(first, 2)) == 2
    assert store.get_history(second, 2) == [{"role": "user", "content": "only second"}]


def test_ingestion_metadata_upsert_and_delete(tmp_path):
    store = ConversationStore(tmp_path / "test.db")
    store.upsert_ingestion_record("facts.txt", "hash1", 4, "now")
    assert store.get_ingestion_record("facts.txt")["chunk_count"] == 4
    store.upsert_ingestion_record("facts.txt", "hash2", 3, "later")
    assert store.get_ingestion_record("facts.txt")["file_hash"] == "hash2"
    store.delete_ingestion_record("facts.txt")
    assert store.get_ingestion_record("facts.txt") is None


def test_message_role_validation(tmp_path):
    store = ConversationStore(tmp_path / "test.db")
    session = store.create_session()
    import pytest

    with pytest.raises(ValueError):
        store.save_message(session, "system", "not an allowed role")
