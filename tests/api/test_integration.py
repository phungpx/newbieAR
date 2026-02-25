import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        with patch("src.api.routers.sessions.BasicRAG") as mock_basic_rag:
            with patch("src.api.routers.sessions.GraphRAG") as mock_graph_rag:
                mock_basic_rag.return_value = MagicMock()
                mock_graph_rag.return_value = MagicMock()
                yield create_app()


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_create_and_delete_session_lifecycle(client):
    # Create
    resp = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "research_papers", "top_k": 5},
    )
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Delete
    resp = await client.delete(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200

    # Chat on deleted session → error SSE event
    resp = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Hello"},
    )
    assert resp.status_code == 200
    assert b"error" in resp.content
