import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_google_vertex_model", return_value=mock_model):
        from src.api.app import create_app
        application = create_app()
    application.state.model = mock_model
    yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_retrieve_vector_returns_results(client):
    from src.models import RetrievalInfo

    mock_results = [
        RetrievalInfo(content="doc content", source="file.pdf - Chunk #1", score=0.9),
        RetrievalInfo(content="other content", source="file.pdf - Chunk #2", score=0.7),
    ]
    with patch("src.api.routers.retrieve.BasicRAG") as MockRAG:
        instance = MagicMock()
        instance.retrieve = AsyncMock(return_value=mock_results)
        MockRAG.return_value = instance
        response = await client.post(
            "/api/v1/retrieve/vector",
            json={"query": "what is docling?", "collection_name": "research_papers", "top_k": 5},
        )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 2
    assert body["results"][0]["content"] == "doc content"
    assert body["results"][0]["score"] == pytest.approx(0.9)


async def test_retrieve_vector_with_rerank(client):
    from src.models import RetrievalInfo

    mock_results = [RetrievalInfo(content="ranked", source="file.pdf - Chunk #1", score=0.95)]
    with patch("src.api.routers.retrieve.BasicRAG") as MockRAG:
        with patch("src.api.routers.retrieve.SentenceTransformersReranker") as MockCE:
            instance = MagicMock()
            instance.retrieve = AsyncMock(return_value=mock_results)
            MockRAG.return_value = instance
            MockCE.return_value = MagicMock()
            response = await client.post(
                "/api/v1/retrieve/vector",
                json={"query": "test", "collection_name": "col", "top_k": 3, "rerank": True},
            )
    assert response.status_code == 200
    # cross_encoder should be set on the instance
    assert instance.cross_encoder is not None


async def test_retrieve_vector_empty_query(client):
    response = await client.post(
        "/api/v1/retrieve/vector",
        json={"query": "", "collection_name": "col"},
    )
    assert response.status_code == 422


async def test_retrieve_graph_returns_contexts(client):
    mock_contexts = ["Node Content:\n- foo (Citation: g1)"]
    mock_citations = ["g1"]
    with patch("src.api.routers.retrieve.GraphRAG") as MockGraph:
        instance = MagicMock()
        instance.retrieve = AsyncMock(return_value=(mock_contexts, mock_citations))
        instance.close = AsyncMock()
        MockGraph.return_value = instance
        response = await client.post(
            "/api/v1/retrieve/graph",
            json={"query": "what is graphiti?", "top_k": 10},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["contexts"] == mock_contexts
    assert body["citations"] == mock_citations
    instance.close.assert_awaited_once()


async def test_retrieve_graph_empty_query(client):
    response = await client.post(
        "/api/v1/retrieve/graph",
        json={"query": ""},
    )
    assert response.status_code == 422
