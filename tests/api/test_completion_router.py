import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from src.api.app import create_app
from src.api.routers.sessions import store as session_store
from src.agents.deps import AgentDependencies


def make_session(collection_name: str = "test", top_k: int = 5) -> str:
    deps = AgentDependencies(
        basic_rag=MagicMock(),
        graph_rag=MagicMock(),
        top_k=top_k,
    )
    return session_store.create(deps=deps, collection_name=collection_name, top_k=top_k)


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        application = create_app()
    application.state.model = mock_model
    yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_completion_session_not_found(client):
    response = await client.post(
        "/api/v1/completion",
        json={"session_id": "nonexistent", "message": "Hello"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


async def test_completion_message_too_long(client):
    session_id = make_session()
    response = await client.post(
        "/api/v1/completion",
        json={"session_id": session_id, "message": "x" * 1001},
    )
    assert response.status_code == 422


async def test_completion_returns_text_contexts_citations(client):
    session_id = make_session()

    mock_result = MagicMock()
    mock_result.data = "Docling is a document conversion library."
    mock_result.all_messages = MagicMock(return_value=[])

    async def fake_run(*args, **kwargs):
        state = session_store.get(session_id)
        state.deps.contexts = ["ctx1"]
        state.deps.citations = ["cite1"]
        return mock_result

    with patch("src.api.routers.completion.agentic_rag.run", new=fake_run):
        response = await client.post(
            "/api/v1/completion",
            json={"session_id": session_id, "message": "What is docling?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Docling is a document conversion library."
    assert data["contexts"] == ["ctx1"]
    assert data["citations"] == ["cite1"]
