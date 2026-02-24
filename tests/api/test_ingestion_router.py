import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.ingestion.router import router
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.api import deps


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def job_store():
    return InMemoryJobStore()


@pytest.fixture
def test_app_with_store(test_app, job_store):
    test_app.dependency_overrides[deps.get_job_store] = lambda: job_store
    yield test_app
    test_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_vectordb_returns_202(test_app_with_store):
    with patch("src.api.ingestion.service.VectorDBIngestion") as mock_cls:
        mock_cls.return_value.ingest_file.return_value = {
            "qdrant_collection_name": "test",
            "file_save_path": "/tmp/a",
            "chunk_save_path": "/tmp/b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=test_app_with_store), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingestion/vectordb",
                data={"collection_name": "test", "chunk_strategy": "hybrid"},
                files={"file": ("doc.pdf", b"PDF content", "application/pdf")},
            )
    assert response.status_code == 202
    assert "job_id" in response.json()


@pytest.mark.asyncio
async def test_get_job_status_not_found(test_app_with_store):
    async with AsyncClient(
        transport=ASGITransport(app=test_app_with_store), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/ingestion/jobs/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_status_found(test_app_with_store, job_store):
    job = job_store.create_job()
    job_store.update_job(job.job_id, JobStatus.DONE, result={"qdrant_collection_name": "test"})
    async with AsyncClient(
        transport=ASGITransport(app=test_app_with_store), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/ingestion/jobs/{job.job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert data["result"]["qdrant_collection_name"] == "test"


@pytest.mark.asyncio
async def test_ingest_graphdb_returns_202(test_app_with_store):
    async def mock_ingest_file(path):
        pass

    async def mock_close():
        pass

    mock_obj = AsyncMock()
    mock_obj.ingest_file = mock_ingest_file
    mock_obj.close = mock_close

    with patch("src.api.ingestion.service.GraphitiIngestion", return_value=mock_obj):
        async with AsyncClient(
            transport=ASGITransport(app=test_app_with_store), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingestion/graphdb",
                data={"chunk_strategy": "hierarchical"},
                files={"file": ("doc.pdf", b"PDF content", "application/pdf")},
            )
    assert response.status_code == 202
    assert "job_id" in response.json()
