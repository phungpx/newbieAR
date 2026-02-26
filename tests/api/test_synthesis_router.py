import pytest
import httpx
from unittest.mock import MagicMock, patch

from src.api.job_store import JobStatus, job_store


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


async def test_create_synthesis_job_returns_202(client):
    with patch("src.api.routers.synthesis.BackgroundTasks.add_task"):
        response = await client.post(
            "/api/v1/synthesis/jobs",
            json={
                "file_dir": "data/papers/files",
                "output_dir": "data/goldens",
                "topic": "paper",
                "num_contexts": 3,
                "context_size": 3,
            },
        )
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"


async def test_get_synthesis_job_pending(client):
    # Pre-populate job store
    job_id = job_store.create()
    response = await client.get(f"/api/v1/synthesis/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["job_id"] == job_id


async def test_get_synthesis_job_done(client):
    job_id = job_store.create()
    job_store.update(job_id, status=JobStatus.DONE, result={"goldens_count": 10, "output_dir": "data/goldens"})
    response = await client.get(f"/api/v1/synthesis/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["result"]["goldens_count"] == 10


async def test_get_synthesis_job_failed(client):
    job_id = job_store.create()
    job_store.update(job_id, status=JobStatus.FAILED, error="something went wrong")
    response = await client.get(f"/api/v1/synthesis/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == "something went wrong"


async def test_get_synthesis_job_not_found(client):
    response = await client.get("/api/v1/synthesis/jobs/nonexistent-id")
    assert response.status_code == 404


async def test_upload_synthesis_files_success(client):
    response = await client.post(
        "/api/v1/synthesis/upload",
        files=[
            ("files", ("paper1.pdf", b"%PDF-1.4 content1", "application/pdf")),
            ("files", ("paper2.pdf", b"%PDF-1.4 content2", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_count"] == 2
    assert "file_dir" in body
    # Cleanup
    import shutil, os
    if os.path.isdir(body["file_dir"]):
        shutil.rmtree(body["file_dir"])


async def test_upload_synthesis_files_empty(client):
    response = await client.post("/api/v1/synthesis/upload", files=[])
    assert response.status_code == 422  # FastAPI validation: files required
