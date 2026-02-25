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


async def test_ingest_vector_success(client):
    mock_result = {
        "file_save_path": "/tmp/test.md",
        "chunk_save_path": "/tmp/chunks/test.json",
        "qdrant_collection_name": "research_papers",
    }
    with patch("src.api.routers.ingest.asyncio.to_thread", return_value=mock_result):
        with patch("src.api.routers.ingest.VectorDBIngestion") as MockVec:
            MockVec.return_value = MagicMock()
            response = await client.post(
                "/api/v1/ingest/vector",
                files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
                data={"collection_name": "research_papers", "chunk_strategy": "hybrid"},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "research_papers"
    assert body["chunk_strategy"] == "hybrid"


async def test_ingest_vector_invalid_chunk_strategy(client):
    with patch("src.api.routers.ingest.VectorDBIngestion") as MockVec:
        MockVec.side_effect = ValueError("Invalid chunk strategy")
        response = await client.post(
            "/api/v1/ingest/vector",
            files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
            data={"collection_name": "research_papers", "chunk_strategy": "invalid"},
        )
    assert response.status_code == 400


async def test_ingest_graph_success(client):
    with patch("src.api.routers.ingest.GraphitiIngestion") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.ingest_file = AsyncMock()
        mock_instance.close = AsyncMock()
        MockGraph.return_value = mock_instance
        response = await client.post(
            "/api/v1/ingest/graph",
            files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
            data={"chunk_strategy": "hierarchical"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_strategy"] == "hierarchical"
    mock_instance.ingest_file.assert_awaited_once()
    mock_instance.close.assert_awaited_once()


async def test_ingest_graph_default_chunk_strategy(client):
    with patch("src.api.routers.ingest.GraphitiIngestion") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.ingest_file = AsyncMock()
        mock_instance.close = AsyncMock()
        MockGraph.return_value = mock_instance
        response = await client.post(
            "/api/v1/ingest/graph",
            files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_strategy"] == "hierarchical"
