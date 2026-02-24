import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.retrieval.router import router
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_retrieve_basic_returns_results(test_app):
    mock_service = MagicMock()
    mock_service.retrieve_basic.return_value = [
        RetrievalInfo(content="doc", source="file.pdf - Chunk #1", score=0.95)
    ]
    with patch("src.api.retrieval.router.RetrievalService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/basic",
                json={"query": "what is RAG?", "collection_name": "papers", "top_k": 3},
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "doc"


@pytest.mark.asyncio
async def test_retrieve_graph_returns_results(test_app):
    mock_service = AsyncMock()
    mock_service.retrieve_graph.return_value = (
        [GraphitiNodeInfo(uuid="n1", summary="Node 1")],
        [GraphitiEdgeInfo(uuid="e1", fact="Fact 1")],
        [GraphitiEpisodeInfo(uuid="ep1", content="Episode 1")],
    )
    with patch("src.api.retrieval.router.RetrievalService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/graph",
                json={"query": "who is the author?", "top_k": 5},
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["uuid"] == "n1"
