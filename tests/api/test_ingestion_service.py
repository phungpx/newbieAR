import asyncio
import pytest
from unittest.mock import patch, MagicMock
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.api.ingestion.service import IngestionService


@pytest.fixture
def job_store():
    return InMemoryJobStore()


@pytest.fixture
def service(job_store):
    return IngestionService(job_store)


@pytest.mark.asyncio
async def test_ingest_vectordb_returns_job_id(service, job_store):
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_file.return_value = {
        "file_save_path": "/tmp/doc.md",
        "chunk_save_path": "/tmp/chunks.json",
        "qdrant_collection_name": "test",
    }
    with patch("src.api.ingestion.service.VectorDBIngestion", return_value=mock_ingestion):
        job_id = await service.ingest_vectordb(
            file_bytes=b"PDF content",
            filename="test.pdf",
            collection_name="test",
            chunk_strategy="hybrid",
        )
        assert job_id is not None
        # Give background task time to run (inside patch context so mock is still active)
        await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.DONE
    assert job.result["qdrant_collection_name"] == "test"


@pytest.mark.asyncio
async def test_ingest_vectordb_marks_failed_on_error(service, job_store):
    with patch("src.api.ingestion.service.VectorDBIngestion", side_effect=Exception("connection error")):
        job_id = await service.ingest_vectordb(
            file_bytes=b"data",
            filename="doc.pdf",
            collection_name="col",
            chunk_strategy="hybrid",
        )
        await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert "connection error" in job.error


@pytest.mark.asyncio
async def test_ingest_graphdb_returns_job_id(service, job_store):
    mock_ingestion = MagicMock()

    async def mock_ingest_file(path):
        pass

    async def mock_close():
        pass

    mock_ingestion.ingest_file = mock_ingest_file
    mock_ingestion.close = mock_close

    with patch("src.api.ingestion.service.GraphitiIngestion", return_value=mock_ingestion):
        job_id = await service.ingest_graphdb(
            file_bytes=b"data",
            filename="doc.pdf",
            chunk_strategy="hierarchical",
        )
        await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.DONE
