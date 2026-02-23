import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from src.api.routers.models import router


@pytest.fixture
def models_app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_list_models_returns_200(models_app):
    async with AsyncClient(
        transport=ASGITransport(app=models_app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_models_contains_both_agents(models_app):
    async with AsyncClient(
        transport=ASGITransport(app=models_app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models")
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "basic-rag" in ids
    assert "graph-rag" in ids


@pytest.mark.asyncio
async def test_list_models_includes_gemini_variants(models_app):
    async with AsyncClient(
        transport=ASGITransport(app=models_app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models")
    ids = [m["id"] for m in resp.json()["data"]]
    assert "basic-rag/gemini-2.5-flash" in ids
    assert "graph-rag/gemini-2.5-flash" in ids
