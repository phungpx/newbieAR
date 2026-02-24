# FastAPI Agentic RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI wrapper around the pydantic-ai agentic RAG agent with SSE streaming and in-memory multi-turn session management.

**Architecture:** Thin `src/api/` layer over the existing `agentic_rag` Agent singleton. Sessions created via `POST /sessions` store `AgentDependencies` + pydantic-ai `messages` history in an in-memory `SessionStore`. Chat messages stream via SSE using `sse-starlette`. The pydantic-ai model is initialized once at app startup in the FastAPI lifespan and stored in `app.state.model`.

**Tech Stack:** FastAPI, uvicorn, sse-starlette, pydantic-ai, httpx + pytest-asyncio (tests — `asyncio_mode = "auto"` already configured in `pyproject.toml`)

---

## File Map (all new unless noted)

```
src/api/
  __init__.py
  app.py               # FastAPI factory + lifespan + CORS
  schemas.py           # Pydantic request/response models
  session_store.py     # In-memory SessionStore + SessionState
  routers/
    __init__.py
    sessions.py        # POST /sessions, DELETE /sessions/{id}
    chat.py            # POST /chat → SSE stream

tests/api/
  __init__.py
  conftest.py          # shared fixtures
  test_schemas.py
  test_session_store.py
  test_sessions_router.py
  test_chat_router.py
  test_integration.py
```

**Settings already available** (`src/settings.py`):
- `settings.api_prefix` → `"/api/v1"` (prefix for all routers)
- `settings.cors_origins` → list of allowed origins
- `settings.llm_model`, `settings.llm_base_url`, `settings.llm_api_key` → model init

**Key imports you will use:**
```python
from src.agents.agentic_rag import agentic_rag        # the Agent singleton
from src.agents.deps import AgentDependencies
from src.agents.models import get_openai_model
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG
from src.settings import settings
from pydantic_ai.messages import ModelMessage
```

---

### Task 1: Schemas

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_schemas.py`

**Step 1: Create empty init files**

```bash
touch src/api/__init__.py tests/api/__init__.py
```

**Step 2: Write the failing tests**

Create `tests/api/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from src.api.schemas import CreateSessionRequest, CreateSessionResponse, DeleteSessionResponse, ChatRequest


def test_create_session_request_valid():
    req = CreateSessionRequest(collection_name="research_papers", top_k=5)
    assert req.collection_name == "research_papers"
    assert req.top_k == 5


def test_create_session_request_default_top_k():
    req = CreateSessionRequest(collection_name="test")
    assert req.top_k == 5


def test_create_session_request_missing_collection_name():
    with pytest.raises(ValidationError):
        CreateSessionRequest()


def test_create_session_response():
    resp = CreateSessionResponse(session_id="abc123", collection_name="test", top_k=5)
    assert resp.session_id == "abc123"


def test_chat_request_valid():
    req = ChatRequest(session_id="abc123", message="What is docling?")
    assert req.message == "What is docling?"


def test_chat_request_message_too_long():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="x" * 1001)


def test_chat_request_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="")


def test_chat_request_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="   ")
```

**Step 3: Run tests to verify failure**

```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.schemas'`

**Step 4: Implement schemas**

Create `src/api/schemas.py`:

```python
from pydantic import BaseModel, field_validator


class CreateSessionRequest(BaseModel):
    collection_name: str
    top_k: int = 5


class CreateSessionResponse(BaseModel):
    session_id: str
    collection_name: str
    top_k: int


class DeleteSessionResponse(BaseModel):
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message must be 1000 characters or fewer")
        return v
```

**Step 5: Run tests to verify pass**

```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: 8 passed

**Step 6: Commit**

```bash
git add src/api/__init__.py src/api/schemas.py tests/api/__init__.py tests/api/test_schemas.py
git commit -m "feat(api): add request/response schemas"
```

---

### Task 2: Session Store

**Files:**
- Create: `src/api/session_store.py`
- Create: `tests/api/test_session_store.py`

**Step 1: Write the failing tests**

Create `tests/api/test_session_store.py`:

```python
from unittest.mock import MagicMock
from src.api.session_store import SessionStore, SessionState
from src.agents.deps import AgentDependencies


def make_deps() -> AgentDependencies:
    return AgentDependencies(
        basic_rag=MagicMock(),
        graph_rag=MagicMock(),
        top_k=5,
    )


def test_create_and_get_session():
    store = SessionStore()
    deps = make_deps()
    session_id = store.create(deps=deps, collection_name="research_papers", top_k=5)
    state = store.get(session_id)
    assert state is not None
    assert state.collection_name == "research_papers"
    assert state.top_k == 5
    assert state.messages == []


def test_get_nonexistent_session():
    store = SessionStore()
    assert store.get("does-not-exist") is None


def test_delete_session():
    store = SessionStore()
    session_id = store.create(deps=make_deps(), collection_name="test", top_k=3)
    store.delete(session_id)
    assert store.get(session_id) is None


def test_delete_nonexistent_session_no_error():
    store = SessionStore()
    store.delete("nonexistent")  # must not raise


def test_session_id_is_unique():
    store = SessionStore()
    id1 = store.create(deps=make_deps(), collection_name="a", top_k=5)
    id2 = store.create(deps=make_deps(), collection_name="b", top_k=5)
    assert id1 != id2
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_session_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.session_store'`

**Step 3: Implement session store**

Create `src/api/session_store.py`:

```python
import uuid
from dataclasses import dataclass, field
from pydantic_ai.messages import ModelMessage
from src.agents.deps import AgentDependencies


@dataclass
class SessionState:
    deps: AgentDependencies
    messages: list[ModelMessage] = field(default_factory=list)
    collection_name: str = ""
    top_k: int = 5


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def create(self, deps: AgentDependencies, collection_name: str, top_k: int) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionState(
            deps=deps,
            collection_name=collection_name,
            top_k=top_k,
        )
        return session_id

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/api/test_session_store.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/api/session_store.py tests/api/test_session_store.py
git commit -m "feat(api): add in-memory session store"
```

---

### Task 3: App Skeleton

**Files:**
- Create: `src/api/routers/__init__.py`
- Create: `src/api/app.py`
- Create: `tests/api/conftest.py`

**Step 1: Write the failing test**

Create `tests/api/conftest.py`:

```python
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        application = create_app()
    return application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
```

Add a health check test to `tests/api/test_schemas.py`:

```python
async def test_app_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_schemas.py::test_app_health -v
```
Expected: `ModuleNotFoundError: No module named 'src.api.app'`

**Step 3: Implement app skeleton**

Create `src/api/routers/__init__.py` (empty).

Create `src/api/app.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.models import get_openai_model
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = get_openai_model(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
```

**Step 4: Run test**

```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: All pass including `test_app_health`

**Step 5: Commit**

```bash
git add src/api/routers/__init__.py src/api/app.py tests/api/conftest.py tests/api/test_schemas.py
git commit -m "feat(api): add FastAPI app skeleton with health endpoint"
```

---

### Task 4: Sessions Router

**Files:**
- Create: `src/api/routers/sessions.py`
- Create: `tests/api/test_sessions_router.py`
- Modify: `src/api/app.py` (add router)

**Step 1: Write the failing tests**

Create `tests/api/test_sessions_router.py`:

```python
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        with patch("src.api.routers.sessions.BasicRAG") as mock_basic_rag:
            with patch("src.api.routers.sessions.GraphRAG") as mock_graph_rag:
                mock_basic_rag.return_value = MagicMock()
                mock_graph_rag.return_value = MagicMock()
                application = create_app()
                yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_create_session(client):
    response = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "research_papers", "top_k": 5},
    )
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["collection_name"] == "research_papers"
    assert data["top_k"] == 5


async def test_create_session_missing_collection_name(client):
    response = await client.post("/api/v1/sessions", json={"top_k": 5})
    assert response.status_code == 422


async def test_delete_session(client):
    create_resp = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "test", "top_k": 3},
    )
    session_id = create_resp.json()["session_id"]

    response = await client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Session deleted"


async def test_delete_nonexistent_session(client):
    response = await client.delete("/api/v1/sessions/nonexistent-id")
    assert response.status_code == 404
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_sessions_router.py -v
```
Expected: 404 on all routes — sessions router not registered yet

**Step 3: Implement sessions router**

Create `src/api/routers/sessions.py`:

```python
from fastapi import APIRouter, HTTPException, status

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG
from src.agents.deps import AgentDependencies
from src.api.schemas import CreateSessionRequest, CreateSessionResponse, DeleteSessionResponse
from src.api.session_store import SessionStore

router = APIRouter(prefix="/sessions", tags=["sessions"])
store = SessionStore()


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    deps = AgentDependencies(
        basic_rag=BasicRAG(qdrant_collection_name=request.collection_name),
        graph_rag=GraphRAG(),
        top_k=request.top_k,
    )
    session_id = store.create(
        deps=deps,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )
    return CreateSessionResponse(
        session_id=session_id,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    if store.get(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    store.delete(session_id)
    return DeleteSessionResponse(message="Session deleted")
```

**Register the router in `src/api/app.py`** — add these two lines inside `create_app()`, after middleware:

```python
# add this import at top of app.py
from src.api.routers.sessions import router as sessions_router

# add this inside create_app(), after add_middleware:
app.include_router(sessions_router, prefix=settings.api_prefix)
```

**Step 4: Run tests**

```bash
uv run pytest tests/api/test_sessions_router.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/api/routers/sessions.py tests/api/test_sessions_router.py src/api/app.py
git commit -m "feat(api): add sessions router (create + delete)"
```

---

### Task 5: Chat Router (SSE)

**Files:**
- Create: `src/api/routers/chat.py`
- Create: `tests/api/test_chat_router.py`
- Modify: `src/api/app.py` (add router)

**Step 1: Write the failing tests**

Create `tests/api/test_chat_router.py`:

```python
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
    state = session_store.get(session_id)
    state.deps.contexts = ["ctx1"]
    state.deps.citations = ["cite1"]

    # Build a mock result that streams two chunks
    mock_result = MagicMock()

    async def fake_stream_text(delta):
        for chunk in ["Hello", " world"]:
            yield chunk

    mock_result.stream_text = fake_stream_text
    mock_result.all_messages = MagicMock(return_value=[])

    class FakeRunStream:
        async def __aenter__(self):
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
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_chat_router.py -v
```
Expected: routes return 404 — chat router not registered yet

**Step 3: Implement chat router**

Create `src/api/routers/chat.py`:

```python
import json
from loguru import logger
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.agents.agentic_rag import agentic_rag
from src.api.schemas import ChatRequest
from src.api.routers.sessions import store as session_store

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    async def event_generator():
        state = session_store.get(body.session_id)
        if state is None:
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Session '{body.session_id}' not found"}),
            }
            return

        state.deps.clear_context()
        model = request.app.state.model

        try:
            async with agentic_rag.run_stream(
                body.message,
                model=model,
                message_history=state.messages,
                deps=state.deps,
            ) as result:
                async for chunk in result.stream_text(delta=True):
                    yield {"event": "delta", "data": json.dumps({"text": chunk})}

            state.messages.extend(result.all_messages())
            yield {
                "event": "done",
                "data": json.dumps({
                    "contexts": state.deps.contexts or [],
                    "citations": state.deps.citations or [],
                }),
            }
        except Exception as e:
            logger.exception(f"Chat stream error: {e}")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return EventSourceResponse(event_generator())
```

**Register the router in `src/api/app.py`** — add these two lines:

```python
# add this import at top of app.py
from src.api.routers.chat import router as chat_router

# add inside create_app(), after sessions_router:
app.include_router(chat_router, prefix=settings.api_prefix)
```

**Step 4: Run tests**

```bash
uv run pytest tests/api/test_chat_router.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/api/routers/chat.py tests/api/test_chat_router.py src/api/app.py
git commit -m "feat(api): add SSE streaming chat endpoint"
```

---

### Task 6: Integration Test & Docs

**Files:**
- Create: `tests/api/test_integration.py`
- Modify: `docs/CLAUDE.md` (add API run command)

**Step 1: Write integration test**

Create `tests/api/test_integration.py`:

```python
import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        with patch("src.api.routers.sessions.BasicRAG") as mock_basic_rag:
            with patch("src.api.routers.sessions.GraphRAG") as mock_graph_rag:
                mock_basic_rag.return_value = MagicMock()
                mock_graph_rag.return_value = MagicMock()
                yield create_app()


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_create_and_delete_session_lifecycle(client):
    # Create
    resp = await client.post(
        "/api/v1/sessions",
        json={"collection_name": "research_papers", "top_k": 5},
    )
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Delete
    resp = await client.delete(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200

    # Chat on deleted session → error SSE event
    resp = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Hello"},
    )
    assert resp.status_code == 200
    assert b"error" in resp.content
```

**Step 2: Run all API tests**

```bash
uv run pytest tests/api/ -v
```
Expected: All tests pass

**Step 3: Update CLAUDE.md**

In `docs/CLAUDE.md`, add to the "CLI Entry Points" section:

```bash
# Run the FastAPI server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

**Step 5: Commit**

```bash
git add tests/api/test_integration.py docs/CLAUDE.md
git commit -m "feat(api): integration tests and docs update"
```

---

## Final Check

After all tasks are complete, verify the server starts without errors (requires `.env` to be configured):

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Expected: server starts, `GET http://localhost:8000/health` returns `{"status": "ok"}`.
