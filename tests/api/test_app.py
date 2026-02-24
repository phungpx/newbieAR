import pytest
from httpx import AsyncClient, ASGITransport
from src.api.app import app


@pytest.mark.asyncio
async def test_docs_accessible():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ingestion_job_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/ingestion/jobs/nonexistent-job")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/v1/agents/sessions/nonexistent")
    assert response.status_code == 404
