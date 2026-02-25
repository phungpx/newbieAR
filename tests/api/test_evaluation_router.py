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


async def test_create_evaluation_job_returns_202(client):
    response = await client.post(
        "/api/v1/evaluation/jobs",
        json={
            "goldens_dir": "data/goldens",
            "collection_name": "research_papers",
            "retrieval_window_size": 5,
            "threshold": 0.5,
            "force_rerun": False,
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"


async def test_get_evaluation_job_pending(client):
    job_id = job_store.create()
    response = await client.get(f"/api/v1/evaluation/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["job_id"] == job_id


async def test_get_evaluation_job_done(client):
    job_id = job_store.create()
    result = {
        "evaluated": 10,
        "skipped": 2,
        "avg_scores": {"AnswerRelevancy": 0.8, "Faithfulness": 0.75},
    }
    job_store.update(job_id, status=JobStatus.DONE, result=result)
    response = await client.get(f"/api/v1/evaluation/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["result"]["evaluated"] == 10
    assert body["result"]["skipped"] == 2
    assert body["result"]["avg_scores"]["AnswerRelevancy"] == pytest.approx(0.8)


async def test_get_evaluation_job_failed(client):
    job_id = job_store.create()
    job_store.update(job_id, status=JobStatus.FAILED, error="evaluation crashed")
    response = await client.get(f"/api/v1/evaluation/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == "evaluation crashed"


async def test_get_evaluation_job_not_found(client):
    response = await client.get("/api/v1/evaluation/jobs/nonexistent-id")
    assert response.status_code == 404
