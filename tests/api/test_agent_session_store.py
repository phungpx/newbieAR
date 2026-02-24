import pytest
from src.api.agents.session_store import InMemorySessionStore


def test_get_or_create_new_session():
    store = InMemorySessionStore()
    session_id, messages = store.get_or_create(None)
    assert session_id is not None
    assert messages == []


def test_get_or_create_existing_session():
    store = InMemorySessionStore()
    session_id, _ = store.get_or_create(None)
    store.save(session_id, [{"role": "user", "content": "hi"}])
    same_id, messages = store.get_or_create(session_id)
    assert same_id == session_id
    assert len(messages) == 1


def test_get_or_create_unknown_session_id_creates_new():
    store = InMemorySessionStore()
    new_id, messages = store.get_or_create("does-not-exist")
    assert new_id != "does-not-exist"
    assert messages == []


def test_delete_existing_session():
    store = InMemorySessionStore()
    session_id, _ = store.get_or_create(None)
    result = store.delete(session_id)
    assert result is True
    _, messages = store.get_or_create(session_id)
    assert messages == []


def test_delete_nonexistent_session_returns_false():
    store = InMemorySessionStore()
    assert store.delete("ghost") is False
