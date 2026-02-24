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
                application = create_app()
                yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_create_session(client):
    response = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "research_papers", "top_k": 5},
    )
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["collection_name"] == "research_papers"
    assert data["top_k"] == 5


async def test_create_session_missing_collection_name(client):
    response = await client.post("/api/v1/sessions", json={"top_k": 5})
    assert response.status_code == 422


async def test_delete_session(client):
    create_resp = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "test", "top_k": 3},
    )
    session_id = create_resp.json()["session_id"]

    response = await client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Session deleted"


async def test_delete_nonexistent_session(client):
    response = await client.delete("/api/v1/sessions/nonexistent-id")
    assert response.status_code == 404
