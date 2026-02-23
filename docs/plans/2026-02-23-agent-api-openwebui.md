# Agent API + Open WebUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose `basic_rag_agent` and `graphiti_agent` as an OpenAI-compatible streaming HTTP API so Open WebUI can interact with both agents as selectable models.

**Architecture:** A new `src/api/` FastAPI package implements `GET /v1/models` and `POST /v1/chat/completions`. Two agents appear as model IDs (`basic-rag`, `graph-rag`). The `model` field in the request encodes both agent type and LLM choice (`agent-type/llm-model`). Multi-turn conversation history is stored in Redis keyed by a SHA-256 derived from `(model + first user message)`. Streaming uses SSE in OpenAI delta format. Open WebUI runs in Docker alongside the API and Redis.

**Tech Stack:** FastAPI, pydantic-ai (agents already built), redis[hiredis] asyncio, httpx + pytest-asyncio + fakeredis for tests, Open WebUI Docker image.

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add redis and test dependencies**

In `pyproject.toml`, add to `dependencies`:
```toml
"redis[hiredis]>=5.0",
```

Add a new optional-dependencies section after `dependencies`:
```toml
[project.optional-dependencies]
test = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
    "fakeredis>=2.26",
]
```

**Step 2: Sync**

```bash
uv sync
```
Expected: resolves and installs redis, no errors.

**Step 3: Verify**
```bash
uv run python -c "import redis.asyncio; print('ok')"
```
Expected: `ok`

**Step 4: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "feat: add redis and test dependencies"
```

---

## Task 2: Add RedisSettings to settings.py

**Files:**
- Modify: `src/settings.py`

**Step 1: Add `RedisSettings` class**

After `class SessionSettings` in `src/settings.py`, add:
```python
class RedisSettings(ProjectBaseSettings):
    redis_url: str = "redis://localhost:6379"
```

**Step 2: Add it to `ProjectSettings` inheritance chain and expose a property**

Change `ProjectSettings` class definition:
```python
class ProjectSettings(
    OpenAILLMSettings,
    LangfuseSettings,
    QdrantVectorStoreSettings,
    OpenAIEmbeddingSettings,
    ConfidentSettings,
    RerankerSettings,
    Neo4jGraphDBSettings,
    MinIOSettings,
    CritiqueModelSettings,
    APISettings,
    AuthSettings,
    JobSettings,
    SessionSettings,
    RedisSettings,        # ← add this
):
```

Add property:
```python
    @property
    def redis(self) -> RedisSettings:
        return RedisSettings(**self.model_dump())
```

**Step 3: Verify settings load**
```bash
uv run python -c "from src.settings import settings; print(settings.redis_url)"
```
Expected: `redis://localhost:6379`

**Step 4: Commit**
```bash
git add src/settings.py
git commit -m "feat: add RedisSettings with redis_url"
```

---

## Task 3: Refactor `src/agents/agentic_basic_rag.py`

The file currently hard-codes `llm_provider = "google"` and instantiates the model at module level. We need to remove this so the API can inject the model dynamically. `pydantic_ai.Agent` accepts `model=None` and allows `model=` override at `run_stream` call time.

We also need to wrap the blocking `basic_rag.generate()` call in `asyncio.to_thread` so it doesn't block the event loop when served via the API.

**Files:**
- Modify: `src/agents/agentic_basic_rag.py`

**Step 1: Remove module-level model code and make agent model-agnostic**

Replace the block:
```python
llm_provider = "google"
if llm_provider == "openai":
    model = get_openai_model()
elif llm_provider == "google":
    model = get_google_model()
else:
    raise ValueError(f"Invalid LLM provider: {llm_provider}")

basic_rag_agent = Agent(
    model=model,
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=BasicRAGDependencies,
    retries=2,
)
```

With:
```python
basic_rag_agent = Agent(
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=BasicRAGDependencies,
    retries=2,
)
```

**Step 2: Wrap sync `basic_rag.generate()` in asyncio.to_thread**

In the `search_basic_rag` tool function, replace:
```python
retrieval_infos, generated_answer = basic_rag.generate(
    query, top_k=ctx.deps.top_k, return_context=True
)
```
With:
```python
retrieval_infos, generated_answer = await asyncio.to_thread(
    basic_rag.generate, query, top_k=ctx.deps.top_k, return_context=True
)
```

**Step 3: Update the CLI `main()` to create its own model**

In `async def main()`, add before `deps = BasicRAGDependencies(...)`:
```python
model = get_openai_model()
```

And update `basic_rag_agent.run_stream(...)` calls to pass `model=model`:
```python
async with basic_rag_agent.run_stream(
    user_input, model=model, message_history=messages, deps=deps
) as result:
```

**Step 4: Verify the module still imports cleanly**
```bash
uv run python -c "from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies; print('ok')"
```
Expected: `ok` (no errors during import)

**Step 5: Commit**
```bash
git add src/agents/agentic_basic_rag.py
git commit -m "refactor: make basic_rag_agent model-agnostic, wrap sync RAG in to_thread"
```

---

## Task 4: Refactor `src/agents/agentic_graph_rag.py`

Same pattern as Task 3.

**Files:**
- Modify: `src/agents/agentic_graph_rag.py`

**Step 1: Remove module-level model code**

Replace:
```python
llm_provider = "google"
if llm_provider == "openai":
    model = get_openai_model()
elif llm_provider == "google":
    model = get_google_model()
else:
    raise ValueError(f"Invalid LLM provider: {llm_provider}")

graphiti_agent = Agent(
    model=model,
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=GraphitiDependencies,
    retries=2,
)
```

With:
```python
graphiti_agent = Agent(
    system_prompt=AGENTIC_RAG_INSTRUCTION,
    deps_type=GraphitiDependencies,
    retries=2,
)
```

**Step 2: Update CLI `main()` to create its own model**

In `async def main()`, add before the loop:
```python
model = get_openai_model()
```

Update `graphiti_agent.run_stream(...)`:
```python
async with graphiti_agent.run_stream(
    user_input, model=model, message_history=messages, deps=deps
) as result:
```

**Step 3: Verify import**
```bash
uv run python -c "from src.agents.agentic_graph_rag import graphiti_agent, GraphitiDependencies; print('ok')"
```
Expected: `ok`

**Step 4: Commit**
```bash
git add src/agents/agentic_graph_rag.py
git commit -m "refactor: make graphiti_agent model-agnostic"
```

---

## Task 5: Create `src/api/schemas.py`

OpenAI-compatible request/response Pydantic models.

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `tests/__init__.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_schemas.py`

**Step 1: Write the failing test**

Create `tests/__init__.py` and `tests/api/__init__.py` as empty files.

Create `tests/api/test_schemas.py`:
```python
import json
from src.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionChunk,
    DeltaContent,
    ChunkChoice,
    ModelsResponse,
    ModelInfo,
)


def test_chat_completion_request_defaults_stream_true():
    req = ChatCompletionRequest(
        model="basic-rag",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert req.stream is True


def test_chat_completion_request_parses_messages():
    data = {
        "model": "basic-rag",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
    }
    req = ChatCompletionRequest(**data)
    assert len(req.messages) == 2
    assert req.messages[0].role == "user"


def test_chunk_serializes_to_openai_format():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc",
        created=1234567890,
        model="basic-rag",
        choices=[ChunkChoice(delta=DeltaContent(content="Hello"))],
    )
    d = chunk.model_dump()
    assert d["object"] == "chat.completion.chunk"
    assert d["choices"][0]["delta"]["content"] == "Hello"
    assert d["choices"][0]["finish_reason"] is None


def test_models_response_structure():
    resp = ModelsResponse(data=[ModelInfo(id="basic-rag"), ModelInfo(id="graph-rag")])
    assert resp.object == "list"
    assert len(resp.data) == 2
```

**Step 2: Run to verify failure**
```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.schemas'`

**Step 3: Create `src/api/__init__.py`**
```python
# empty
```

**Step 4: Create `src/api/schemas.py`**
```python
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "newbie-ar"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


class CompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class CompletionChoice(BaseModel):
    index: int = 0
    message: CompletionMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo = UsageInfo()
```

**Step 5: Run tests**
```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: all 4 tests PASS

**Step 6: Commit**
```bash
git add src/api/__init__.py src/api/schemas.py tests/__init__.py tests/api/__init__.py tests/api/test_schemas.py
git commit -m "feat: add OpenAI-compatible request/response schemas"
```

---

## Task 6: Create `src/api/services/session.py`

Redis-backed pydantic-ai message history store.

**Files:**
- Create: `src/api/services/__init__.py`
- Create: `src/api/services/session.py`
- Create: `tests/api/test_session.py`

**Step 1: Write the failing tests**

Create `tests/api/test_session.py`:
```python
import pytest
import fakeredis.aioredis as fakeredis
from pydantic_ai.messages import ModelRequest, UserPromptPart

from src.api.services.session import derive_session_key, load_messages, save_messages
from src.api.schemas import ChatMessage


@pytest.mark.asyncio
async def test_load_returns_empty_list_for_new_session():
    redis = fakeredis.FakeRedis()
    messages = await load_messages(redis, "session:nonexistent")
    assert messages == []


@pytest.mark.asyncio
async def test_save_and_load_roundtrip():
    redis = fakeredis.FakeRedis()
    original = [ModelRequest(parts=[UserPromptPart(content="test")])]
    await save_messages(redis, "session:test", original, ttl=60)
    loaded = await load_messages(redis, "session:test")
    assert len(loaded) == 1
    assert loaded[0].parts[0].content == "test"


def test_derive_session_key_stable_across_turns():
    msgs_turn1 = [ChatMessage(role="user", content="What is RAG?")]
    msgs_turn2 = [
        ChatMessage(role="user", content="What is RAG?"),
        ChatMessage(role="assistant", content="RAG is..."),
        ChatMessage(role="user", content="Tell me more"),
    ]
    key1 = derive_session_key("basic-rag", msgs_turn1)
    key2 = derive_session_key("basic-rag", msgs_turn2)
    assert key1 == key2  # same first user message → same session


def test_derive_session_key_differs_by_model():
    msgs = [ChatMessage(role="user", content="Hello")]
    k1 = derive_session_key("basic-rag", msgs)
    k2 = derive_session_key("graph-rag", msgs)
    assert k1 != k2


def test_derive_session_key_has_prefix():
    msgs = [ChatMessage(role="user", content="Hello")]
    key = derive_session_key("basic-rag", msgs)
    assert key.startswith("session:")
```

**Step 2: Run to verify failure**
```bash
uv run pytest tests/api/test_session.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.services'`

**Step 3: Create `src/api/services/__init__.py`** (empty)

**Step 4: Create `src/api/services/session.py`**
```python
from __future__ import annotations

import hashlib

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from redis.asyncio import Redis

from src.api.schemas import ChatMessage

_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(list[ModelMessage])

SESSION_PREFIX = "session:"
DEFAULT_TTL_SECONDS = 86400  # 24 h


def derive_session_key(model: str, messages: list[ChatMessage]) -> str:
    """Stable session key: SHA-256 of (model + first user message content).

    Open WebUI always sends the full message history, so we anchor the key to
    the very first user message which never changes across turns.
    """
    first_user_content = next(
        (m.content for m in messages if m.role == "user"), ""
    )
    raw = f"{model}:{first_user_content}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{SESSION_PREFIX}{digest}"


async def load_messages(redis: Redis, key: str) -> list[ModelMessage]:
    raw = await redis.get(key)
    if not raw:
        return []
    return _adapter.validate_json(raw)


async def save_messages(
    redis: Redis,
    key: str,
    messages: list[ModelMessage],
    ttl: int = DEFAULT_TTL_SECONDS,
) -> None:
    await redis.set(key, _adapter.dump_json(messages), ex=ttl)
```

**Step 5: Run tests**
```bash
uv run pytest tests/api/test_session.py -v
```
Expected: all 5 tests PASS

**Step 6: Commit**
```bash
git add src/api/services/__init__.py src/api/services/session.py tests/api/test_session.py
git commit -m "feat: add Redis session store for pydantic-ai message history"
```

---

## Task 7: Create `src/api/services/agent_runner.py`

Model routing (`parse_model_id`, `make_llm_model`) and SSE streaming generator.

**Files:**
- Create: `src/api/services/agent_runner.py`
- Create: `tests/api/test_agent_runner.py`

**Step 1: Write the failing tests (pure unit tests — no I/O)**

Create `tests/api/test_agent_runner.py`:
```python
import pytest
from unittest.mock import MagicMock
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel

from src.api.services.agent_runner import parse_model_id, make_llm_model, SUPPORTED_AGENTS


def test_parse_model_id_agent_only():
    agent_type, llm = parse_model_id("basic-rag")
    assert agent_type == "basic-rag"
    assert llm is None


def test_parse_model_id_with_llm():
    agent_type, llm = parse_model_id("graph-rag/gemini-2.5-flash")
    assert agent_type == "graph-rag"
    assert llm == "gemini-2.5-flash"


def test_parse_model_id_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        parse_model_id("unknown-agent")


def test_parse_model_id_unknown_agent_with_slash_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        parse_model_id("bad-agent/gpt-4o")


def test_make_llm_model_returns_openai_for_default():
    model = make_llm_model(None)
    assert isinstance(model, OpenAIChatModel)


def test_make_llm_model_returns_openai_for_non_gemini():
    model = make_llm_model("gpt-4o")
    assert isinstance(model, OpenAIChatModel)


def test_make_llm_model_returns_google_for_gemini():
    model = make_llm_model("gemini-2.5-flash")
    assert isinstance(model, GoogleModel)


def test_supported_agents_contains_both():
    assert "basic-rag" in SUPPORTED_AGENTS
    assert "graph-rag" in SUPPORTED_AGENTS
```

**Step 2: Run to verify failure**
```bash
uv run pytest tests/api/test_agent_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.services.agent_runner'`

**Step 3: Create `src/api/services/agent_runner.py`**
```python
from __future__ import annotations

import json
import os
import time
import uuid
import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Awaitable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings

from src.settings import settings
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from src.agents.agentic_graph_rag import graphiti_agent, GraphitiDependencies

SUPPORTED_AGENTS: set[str] = {"basic-rag", "graph-rag"}

KNOWN_MODELS: list[str] = [
    "basic-rag",
    "graph-rag",
    "basic-rag/gemini-2.5-flash",
    "graph-rag/gemini-2.5-flash",
]


def parse_model_id(model_id: str) -> tuple[str, str | None]:
    """Split 'agent-type' or 'agent-type/llm-model' → (agent_type, llm_model).

    Raises ValueError for unknown agent types.
    """
    parts = model_id.split("/", 1)
    agent_type = parts[0]
    llm_model = parts[1] if len(parts) > 1 else None
    if agent_type not in SUPPORTED_AGENTS:
        raise ValueError(
            f"Unknown agent: {agent_type!r}. Must be one of {sorted(SUPPORTED_AGENTS)}"
        )
    return agent_type, llm_model


def make_llm_model(llm_model_name: str | None):
    """Return a pydantic-ai model instance for the given LLM name.

    - None or any non-gemini name → OpenAIChatModel using settings
    - "gemini-*" → GoogleModel via Vertex AI
    """
    model_settings = ModelSettings(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    if llm_model_name and llm_model_name.startswith("gemini-"):
        project = os.getenv("GOOGLE_VERTEX_PROJECT", "vns-durian-traceability")
        return GoogleModel(
            model_name=llm_model_name,
            provider=GoogleProvider(project=project, vertexai=True),
            settings=model_settings,
        )
    return OpenAIChatModel(
        model_name=llm_model_name or settings.llm_model,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        ),
        settings=model_settings,
    )


async def stream_response(
    agent_type: str,
    llm_model_name: str | None,
    user_input: str,
    prior_messages: list[ModelMessage],
    basic_rag,       # BasicRAG instance from app.state
    graph_retrieval, # GraphRetrieval instance from app.state
    session_saver: Callable[[list[ModelMessage]], Awaitable[None]],
) -> AsyncGenerator[str, None]:
    """Yield SSE text chunks in OpenAI delta format, then save the session."""
    model = make_llm_model(llm_model_name)
    model_id = f"{agent_type}/{llm_model_name or settings.llm_model}"
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # First chunk announces the assistant role
    yield _chunk({"role": "assistant", "content": ""})

    if agent_type == "basic-rag":
        agent: Agent = basic_rag_agent
        deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=5)
    else:  # "graph-rag"
        agent = graphiti_agent
        deps = GraphitiDependencies(graph_retrieval=graph_retrieval, top_k=5)

    new_messages: list[ModelMessage] = []
    async with agent.run_stream(
        user_input,
        model=model,
        message_history=prior_messages,
        deps=deps,
    ) as result:
        async for delta in result.stream_text(delta=True):
            yield _chunk({"content": delta})
        new_messages = result.all_messages()

    # Persist the updated history before sending the terminal chunks
    await session_saver(prior_messages + new_messages)

    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"
```

**Step 4: Run tests**
```bash
uv run pytest tests/api/test_agent_runner.py -v
```
Expected: all 8 tests PASS

**Step 5: Commit**
```bash
git add src/api/services/agent_runner.py tests/api/test_agent_runner.py
git commit -m "feat: add agent runner with model routing and SSE streaming"
```

---

## Task 8: Create `src/api/routers/models.py`

`GET /v1/models` — returns the list of supported agent+model combinations.

**Files:**
- Create: `src/api/routers/__init__.py`
- Create: `src/api/routers/models.py`
- Create: `tests/api/test_routes_models.py`

**Step 1: Write the failing test**

Create `tests/api/test_routes_models.py`:
```python
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
```

**Step 2: Run to verify failure**
```bash
uv run pytest tests/api/test_routes_models.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Create `src/api/routers/__init__.py`** (empty)

**Step 4: Create `src/api/routers/models.py`**
```python
import time

from fastapi import APIRouter

from src.api.schemas import ModelInfo, ModelsResponse
from src.api.services.agent_runner import KNOWN_MODELS

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    now = int(time.time())
    return ModelsResponse(
        data=[ModelInfo(id=m, created=now) for m in KNOWN_MODELS]
    )
```

**Step 5: Run tests**
```bash
uv run pytest tests/api/test_routes_models.py -v
```
Expected: all 3 tests PASS

**Step 6: Commit**
```bash
git add src/api/routers/__init__.py src/api/routers/models.py tests/api/test_routes_models.py
git commit -m "feat: add GET /v1/models router"
```

---

## Task 9: Create `src/api/routers/chat.py`

`POST /v1/chat/completions` — wires session + agent runner into a streaming SSE response.

**Files:**
- Create: `src/api/routers/chat.py`
- Create: `tests/api/test_routes_chat.py`

**Step 1: Write the failing tests**

Create `tests/api/test_routes_chat.py`:
```python
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
```

**Step 2: Run to verify failure**
```bash
uv run pytest tests/api/test_routes_chat.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Create `src/api/routers/chat.py`**
```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatCompletionRequest
from src.api.services.agent_runner import parse_model_id, stream_response
from src.api.services.session import derive_session_key, load_messages, save_messages

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    try:
        agent_type, llm_model = parse_model_id(body.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    redis = request.app.state.redis
    session_key = derive_session_key(body.model, body.messages)
    prior_messages = await load_messages(redis, session_key)
    user_input = body.messages[-1].content
    ttl: int = request.app.state.session_ttl

    async def saver(messages):
        await save_messages(redis, session_key, messages, ttl=ttl)

    return StreamingResponse(
        stream_response(
            agent_type=agent_type,
            llm_model_name=llm_model,
            user_input=user_input,
            prior_messages=prior_messages,
            basic_rag=request.app.state.basic_rag,
            graph_retrieval=request.app.state.graph_retrieval,
            session_saver=saver,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Step 4: Run tests**
```bash
uv run pytest tests/api/test_routes_chat.py -v
```
Expected: all 3 tests PASS

**Step 5: Commit**
```bash
git add src/api/routers/chat.py tests/api/test_routes_chat.py
git commit -m "feat: add POST /v1/chat/completions streaming router"
```

---

## Task 10: Create `src/api/app.py`

FastAPI app with lifespan managing Redis, BasicRAG, and GraphRetrieval instances.

**Files:**
- Create: `src/api/app.py`

**Step 1: Create `src/api/app.py`**
```python
from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval
from src.settings import settings
from src.api.routers import chat, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=False
    )
    app.state.basic_rag = BasicRAG()
    app.state.graph_retrieval = GraphRetrieval()
    app.state.session_ttl = settings.session_ttl_hours * 3600
    yield
    # Shutdown
    await app.state.redis.aclose()
    await app.state.graph_retrieval.close()


app = FastAPI(title="newbieAR Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(chat.router)
```

**Step 2: Verify the module is importable**
```bash
uv run python -c "from src.api.app import app; print(app.title)"
```
Expected: `newbieAR Agent API`

**Step 3: Run the full test suite**
```bash
uv run pytest tests/ -v
```
Expected: all tests PASS

**Step 4: Commit**
```bash
git add src/api/app.py
git commit -m "feat: add FastAPI app with lifespan for Redis, BasicRAG, GraphRetrieval"
```

---

## Task 11: Dockerfile.api + docker-compose.openwebui.yaml

**Files:**
- Create: `Dockerfile.api`
- Create: `infras/docker-compose.openwebui.yaml`

**Step 1: Create `Dockerfile.api`**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies (no dev/test extras)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy application source
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create `infras/docker-compose.openwebui.yaml`**
```yaml
services:
  newbie-ar-api:
    build:
      context: ..
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    env_file:
      - ../.env
    environment:
      - REDIS_URL=redis://redis:6379
    volumes:
      # Mount Google credentials for Vertex AI (if using gemini- models)
      - ../google_credentials.json:/app/google_credentials.json:ro
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - newbie-ar-net

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - newbie-ar-net

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://newbie-ar-api:8000/v1
      - OPENAI_API_KEY=empty
      - WEBUI_AUTH=false
    volumes:
      - openwebui_data:/app/backend/data
    depends_on:
      - newbie-ar-api
    restart: unless-stopped
    networks:
      - newbie-ar-net

volumes:
  redis_data:
  openwebui_data:

networks:
  newbie-ar-net:
    driver: bridge
```

**Step 3: Add `REDIS_URL` env var to `.env` for local dev (if you have one)**

Add to your `.env`:
```
REDIS_URL=redis://localhost:6379
```

**Step 4: Update `src/settings.py` `RedisSettings` to use the env var name**

The `redis_url` field in `RedisSettings` reads from `REDIS_URL` in `.env`. Verify the env var name matches by updating the default:
```python
class RedisSettings(ProjectBaseSettings):
    redis_url: str = "redis://localhost:6379"
```
`pydantic_settings` maps `redis_url` → env var `REDIS_URL` automatically.

**Step 5: Verify Dockerfile builds**
```bash
docker build -f Dockerfile.api -t newbie-ar-api:local .
```
Expected: image builds successfully

**Step 6: Verify stack starts**
```bash
# Start backend infrastructure first (Qdrant, Neo4j)
docker compose -f infras/docker-compose.qdrant.yaml up -d
docker compose -f infras/docker-compose.neo4j.yaml up -d

# Start the API + Open WebUI
docker compose -f infras/docker-compose.openwebui.yaml up -d

# Check services are healthy
docker compose -f infras/docker-compose.openwebui.yaml ps
```
Expected: `newbie-ar-api`, `redis`, `open-webui` all show as running.

**Step 7: Smoke test the API**
```bash
curl http://localhost:8000/v1/models
```
Expected:
```json
{"object":"list","data":[{"id":"basic-rag",...},{"id":"graph-rag",...},...]}
```

**Step 8: Open WebUI**

Navigate to `http://localhost:3000`. In the model picker, you should see `basic-rag`, `graph-rag`, `basic-rag/gemini-2.5-flash`, `graph-rag/gemini-2.5-flash`.

**Step 9: Commit**
```bash
git add Dockerfile.api infras/docker-compose.openwebui.yaml
git commit -m "feat: add Dockerfile.api and docker-compose for Open WebUI + Redis"
```

---

## Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add the new API commands section to CLAUDE.md**

Add to the `Common Commands` section:
```bash
# Start the agent API + Open WebUI + Redis
docker compose -f infras/docker-compose.openwebui.yaml up -d

# Run API locally (requires Redis running)
uv run uvicorn src.api.app:app --reload

# Run tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/api/test_session.py -v
```

Add to the `Source Layout` table:
```
| `src/api/` | OpenAI-compatible HTTP API: `app.py` (lifespan), `routers/` (models + chat), `services/` (session + agent_runner) |
| `infras/docker-compose.openwebui.yaml` | Open WebUI + newbie-ar-api + Redis stack |
```

**Step 2: Commit**
```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with API commands and structure"
```
