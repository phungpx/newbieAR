import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.agents.router import router
from src.api.agents.session_store import InMemorySessionStore
from src.api import deps


@pytest.fixture
def session_store():
    return InMemorySessionStore()


@pytest.fixture
def test_app(session_store):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_session_store] = lambda: session_store
    return app


@pytest.mark.asyncio
async def test_agent_basic_streams_sse(test_app):
    async def fake_stream(*args, **kwargs):
        yield 'data: {"type": "token", "content": "Hello"}\n\n'
        yield 'data: {"type": "done", "session_id": "abc", "citations": []}\n\n'

    mock_service = MagicMock()
    mock_service.stream_basic.return_value = fake_stream()

    with patch("src.api.agents.router.AgentService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/agents/basic",
                json={"query": "what is RAG?", "collection_name": "papers"},
            )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_delete_session_success(test_app, session_store):
    session_id, _ = session_store.get_or_create(None)
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/api/v1/agents/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


@pytest.mark.asyncio
async def test_delete_session_not_found(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete("/api/v1/agents/sessions/ghost-id")
    assert response.status_code == 404
