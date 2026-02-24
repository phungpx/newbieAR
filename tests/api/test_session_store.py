from unittest.mock import MagicMock
from src.api.session_store import SessionStore, SessionState
from src.agents.deps import AgentDependencies


def make_deps() -> AgentDependencies:
    return AgentDependencies(
        basic_rag=MagicMock(),
        graph_rag=MagicMock(),
        top_k=5,
    )


def test_create_and_get_session():
    store = SessionStore()
    deps = make_deps()
    session_id = store.create(deps=deps, collection_name="research_papers", top_k=5)
    state = store.get(session_id)
    assert state is not None
    assert state.collection_name == "research_papers"
    assert state.top_k == 5
    assert state.messages == []


def test_get_nonexistent_session():
    store = SessionStore()
    assert store.get("does-not-exist") is None


def test_delete_session():
    store = SessionStore()
    session_id = store.create(deps=make_deps(), collection_name="test", top_k=3)
    store.delete(session_id)
    assert store.get(session_id) is None


def test_delete_nonexistent_session_no_error():
    store = SessionStore()
    store.delete("nonexistent")  # must not raise


def test_session_id_is_unique():
    store = SessionStore()
    id1 = store.create(deps=make_deps(), collection_name="a", top_k=5)
    id2 = store.create(deps=make_deps(), collection_name="b", top_k=5)
    assert id1 != id2
