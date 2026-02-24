import json
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app
from src.api.routers.sessions import store as session_store
from src.agents.deps import AgentDependencies


def make_session(collection_name: str = "test", top_k: int = 5) -> str:
    """Create a session directly in the store (bypasses HTTP for test setup)."""
    deps = AgentDependencies(
        basic_rag=MagicMock(),
        graph_rag=MagicMock(),
        top_k=top_k,
    )
    return session_store.create(deps=deps, collection_name=collection_name, top_k=top_k)


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        application = create_app()
    # ASGITransport doesn't trigger the lifespan, so set the model directly
    application.state.model = mock_model
    yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_chat_session_not_found(client):
    response = await client.post(
        "/api/v1/chat",
        json={"session_id": "nonexistent", "message": "Hello"},
    )
    assert response.status_code == 200
    assert b"error" in response.content


async def test_chat_message_too_long(client):
    session_id = make_session()
    response = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "x" * 1001},
    )
    assert response.status_code == 422


async def test_chat_streams_delta_and_done(client):
    session_id = make_session()

    # Build a mock result that streams two chunks
    mock_result = MagicMock()

    async def fake_stream_text(delta):
        for chunk in ["Hello", " world"]:
            yield chunk

    mock_result.stream_text = fake_stream_text
    mock_result.all_messages = MagicMock(return_value=[])

    class FakeRunStream:
        async def __aenter__(self):
            # Simulate agent tools populating contexts/citations during the run
            state = session_store.get(session_id)
            state.deps.contexts = ["ctx1"]
            state.deps.citations = ["cite1"]
            return mock_result

        async def __aexit__(self, *args):
            pass

    with patch(
        "src.api.routers.chat.agentic_rag.run_stream",
        return_value=FakeRunStream(),
    ):
        async with client.stream(
            "POST",
            "/api/v1/chat",
            json={"session_id": session_id, "message": "What is docling?"},
        ) as response:
            assert response.status_code == 200
            raw = await response.aread()

    events = _parse_sse(raw.decode())
    event_types = [e["event"] for e in events]
    assert "delta" in event_types
    assert "done" in event_types

    done_event = next(e for e in events if e["event"] == "done")
    data = json.loads(done_event["data"])
    assert data["contexts"] == ["ctx1"]
    assert data["citations"] == ["cite1"]


def _parse_sse(text: str) -> list[dict]:
    events, current = [], {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events
