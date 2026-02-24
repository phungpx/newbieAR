import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.agents.session_store import InMemorySessionStore
from src.api.agents.service import AgentService


@pytest.fixture
def session_store():
    return InMemorySessionStore()


@pytest.fixture
def service(session_store):
    return AgentService(session_store)


@pytest.mark.asyncio
async def test_stream_basic_yields_tokens_and_done(service):
    mock_result = MagicMock()
    mock_result.stream_text = MagicMock(return_value=_async_iter(["Hello", " world"]))
    mock_result.all_messages = MagicMock(return_value=[])

    async def mock_run_stream(*args, **kwargs):
        class Ctx:
            async def __aenter__(self):
                return mock_result
            async def __aexit__(self, *a):
                pass
        return Ctx()

    mock_agent = MagicMock()
    mock_agent.run_stream = mock_run_stream

    mock_rag = MagicMock()

    with patch("src.api.agents.service.basic_rag_agent", mock_agent), \
         patch("src.api.agents.service.BasicRAG", return_value=mock_rag), \
         patch("src.api.agents.service.get_openai_model", return_value=MagicMock()):
        chunks = []
        async for chunk in service.stream_basic("what is RAG?", "papers", 3, None):
            chunks.append(chunk)

    events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]
    assert len(token_events) == 2
    assert token_events[0]["content"] == "Hello"
    assert len(done_events) == 1
    assert "session_id" in done_events[0]


@pytest.mark.asyncio
async def test_stream_basic_yields_error_on_exception(service):
    async def bad_run_stream(*args, **kwargs):
        class Ctx:
            async def __aenter__(self):
                raise RuntimeError("model down")
            async def __aexit__(self, *a):
                pass
        return Ctx()

    mock_agent = MagicMock()
    mock_agent.run_stream = bad_run_stream

    with patch("src.api.agents.service.basic_rag_agent", mock_agent), \
         patch("src.api.agents.service.BasicRAG", return_value=MagicMock()), \
         patch("src.api.agents.service.get_openai_model", return_value=MagicMock()):
        chunks = []
        async for chunk in service.stream_basic("q", "col", 3, None):
            chunks.append(chunk)

    events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
    assert any(e["type"] == "error" for e in events)


async def _async_iter(items):
    for item in items:
        yield item
