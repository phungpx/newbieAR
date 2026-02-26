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
        "chunk_count": 3,
        "chunks": [
            {"chunk_id": 0, "text_tokens": 100, "text_preview": "First chunk..."},
            {"chunk_id": 1, "text_tokens": 150, "text_preview": "Second chunk..."},
            {"chunk_id": 2, "text_tokens": 120, "text_preview": "Third chunk..."},
        ],
    }
    with patch("src.api.routers.ingest.asyncio.to_thread", return_value=mock_result):
        with patch("src.api.routers.ingest.VectorDBIngestion") as MockVec:
            MockVec.return_value = MagicMock()
            response = await client.post(
                "/api/v1/ingest/vector",
                files={
                    "file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")
                },
                data={"collection_name": "research_papers", "chunk_strategy": "hybrid"},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "research_papers"
    assert body["chunk_strategy"] == "hybrid"
    assert body["chunk_count"] == 3
    assert len(body["chunks"]) == 3
    assert body["chunks"][0]["chunk_id"] == 0
    assert body["chunks"][0]["text_preview"] == "First chunk..."


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
    mock_result = {
        "filename": "test.pdf",
        "chunk_count": 2,
        "chunks": [
            {"chunk_id": 0, "text_tokens": 80, "text_preview": "Graph chunk one..."},
            {"chunk_id": 1, "text_tokens": 95, "text_preview": "Graph chunk two..."},
        ],
    }
    with patch("src.api.routers.ingest.GraphitiIngestion") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.ingest_file = AsyncMock(return_value=mock_result)
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
    assert body["filename"] == "test.pdf"
    assert body["chunk_count"] == 2
    assert len(body["chunks"]) == 2
    mock_instance.ingest_file.assert_awaited_once()
    mock_instance.close.assert_awaited_once()


async def test_ingest_graph_default_chunk_strategy(client):
    mock_result = {"filename": "test.pdf", "chunk_count": 0, "chunks": []}
    with patch("src.api.routers.ingest.GraphitiIngestion") as MockGraph:
        mock_instance = MagicMock()
        mock_instance.ingest_file = AsyncMock(return_value=mock_result)
        mock_instance.close = AsyncMock()
        MockGraph.return_value = mock_instance
        response = await client.post(
            "/api/v1/ingest/graph",
            files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_strategy"] == "hierarchical"


async def test_get_graph_summary_success(client):
    mock_stats = {"nodes": 142, "relationships": 89, "communities": 7}
    with patch(
        "src.api.routers.ingest.get_neo4j_stats",
        new_callable=AsyncMock,
        return_value=mock_stats,
    ):
        response = await client.get("/api/v1/ingest/graph/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == 142
    assert body["relationships"] == 89
    assert body["communities"] == 7


async def test_get_graph_summary_connection_error(client):
    with patch(
        "src.api.routers.ingest.get_neo4j_stats",
        new_callable=AsyncMock,
        side_effect=Exception("Neo4j unavailable"),
    ):
        response = await client.get("/api/v1/ingest/graph/summary")
    assert response.status_code == 503


async def test_get_collection_info_found(client):
    mock_info = {
        "vectors_count": 42,
        "dimensions": 1536,
        "distance": "cosine",
        "status": "green",
    }
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.get_collection_info.return_value = mock_info
        response = await client.get("/api/v1/ingest/collections/research_papers")
    assert response.status_code == 200
    body = response.json()
    assert body["vectors_count"] == 42
    assert body["dimensions"] == 1536
    assert body["distance"] == "cosine"


async def test_get_collection_info_not_found(client):
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.get_collection_info.return_value = None
        response = await client.get("/api/v1/ingest/collections/nonexistent")
    assert response.status_code == 404


async def test_delete_collection_success(client):
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.delete_collection.return_value = None
        response = await client.delete("/api/v1/ingest/collections/research_papers")
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] == "research_papers"


async def test_delete_collection_failure(client):
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.delete_collection.side_effect = RuntimeError("Failed")
        response = await client.delete("/api/v1/ingest/collections/research_papers")
    assert response.status_code == 500


async def test_clear_graph_success(client):
    with patch(
        "src.api.routers.ingest.clear_data", new_callable=AsyncMock
    ) as mock_clear:
        with patch("src.api.routers.ingest.GraphitiClient") as MockClient:
            mock_driver = MagicMock()
            MockClient.return_value.driver = mock_driver
            MockClient.return_value.close = AsyncMock()
            response = await client.post("/api/v1/ingest/graph/clear")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cleared"
    mock_clear.assert_awaited_once_with(mock_driver)


async def test_clear_graph_failure(client):
    with patch(
        "src.api.routers.ingest.clear_data",
        new_callable=AsyncMock,
        side_effect=Exception("Neo4j error"),
    ):
        with patch("src.api.routers.ingest.GraphitiClient") as MockClient:
            MockClient.return_value.driver = MagicMock()
            MockClient.return_value.close = AsyncMock()
            response = await client.post("/api/v1/ingest/graph/clear")
    assert response.status_code == 500
