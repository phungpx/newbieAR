import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from src.api.routers.chat import router


async def _fake_stream(*args, **kwargs):
    yield 'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"basic-rag","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"basic-rag","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"basic-rag","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    yield "data: [DONE]\n\n"


@pytest.fixture
def chat_app():
    app = FastAPI()
    app.include_router(router)
    # Provide app.state dependencies expected by the router
    app.state.redis = AsyncMock()
    app.state.redis.get = AsyncMock(return_value=None)
    app.state.redis.set = AsyncMock()
    app.state.basic_rag = MagicMock()
    app.state.graph_retrieval = AsyncMock()
    app.state.session_ttl = 86400
    return app


@pytest.mark.asyncio
async def test_chat_returns_400_for_unknown_model(chat_app):
    body = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    async with AsyncClient(
        transport=ASGITransport(app=chat_app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert "Unknown agent" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chat_streams_sse_for_basic_rag(chat_app):
    body = {
        "model": "basic-rag",
        "messages": [{"role": "user", "content": "What is RAG?"}],
        "stream": True,
    }
    with patch(
        "src.api.routers.chat.stream_response", side_effect=_fake_stream
    ):
        async with AsyncClient(
            transport=ASGITransport(app=chat_app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: [DONE]" in resp.text


@pytest.mark.asyncio
async def test_chat_extracts_last_user_message(chat_app):
    """Confirm the router passes only the new user turn to stream_response."""
    body = {
        "model": "basic-rag",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ],
        "stream": True,
    }
    captured_input = {}

    async def capturing_stream(agent_type, llm_model_name, user_input, *args, **kwargs):
        captured_input["user_input"] = user_input
        yield "data: [DONE]\n\n"

    with patch("src.api.routers.chat.stream_response", side_effect=capturing_stream):
        async with AsyncClient(
            transport=ASGITransport(app=chat_app), base_url="http://test"
        ) as client:
            await client.post("/v1/chat/completions", json=body)

    assert captured_input["user_input"] == "Second question"
