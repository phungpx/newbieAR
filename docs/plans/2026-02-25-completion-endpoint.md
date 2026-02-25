# Completion Endpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `POST /api/v1/completion` — a non-streaming endpoint that runs the agentic RAG agent and returns the full text, contexts, and citations in a single JSON response.

**Architecture:** Session-based (requires a pre-created `session_id`). Uses pydantic-ai's non-streaming `Agent.run()` instead of `run_stream()`. Shares the same `SessionStore` singleton from `sessions.py`. New router in `src/api/routers/completion.py` registered in `app.py`.

**Tech Stack:** FastAPI, pydantic-ai (`Agent.run()`), pydantic (response model)

---

## File Map

```
src/api/
  schemas.py              ← modify: add CompletionResponse
  app.py                  ← modify: register completion router
  routers/
    completion.py         ← create

tests/api/
  test_completion_router.py  ← create
```

**Key imports you will use:**
```python
from src.agents.agentic_rag import agentic_rag
from src.api.schemas import ChatRequest, CompletionResponse
from src.api.routers.sessions import store as session_store
from src.api.app import create_app
```

---

### Task 1: CompletionResponse Schema

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `tests/api/test_schemas.py`

**Step 1: Write the failing test**

Add to `tests/api/test_schemas.py`:

```python
from src.api.schemas import CreateSessionRequest, CreateSessionResponse, DeleteSessionResponse, ChatRequest, CompletionResponse


def test_completion_response():
    resp = CompletionResponse(
        text="Docling is a document conversion library.",
        contexts=["ctx1", "ctx2"],
        citations=["cite1"],
    )
    assert resp.text == "Docling is a document conversion library."
    assert resp.contexts == ["ctx1", "ctx2"]
    assert resp.citations == ["cite1"]


def test_completion_response_empty_lists():
    resp = CompletionResponse(text="answer", contexts=[], citations=[])
    assert resp.contexts == []
    assert resp.citations == []
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_schemas.py::test_completion_response tests/api/test_schemas.py::test_completion_response_empty_lists -v
```
Expected: `ImportError: cannot import name 'CompletionResponse'`

**Step 3: Add CompletionResponse to schemas**

In `src/api/schemas.py`, append:

```python
class CompletionResponse(BaseModel):
    text: str
    contexts: list[str]
    citations: list[str]
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/api/test_schemas.py -v
```
Expected: All pass (previously 9, now 11)

**Step 5: Commit**

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat(api): add CompletionResponse schema"
```

---

### Task 2: Completion Router

**Files:**
- Create: `src/api/routers/completion.py`
- Create: `tests/api/test_completion_router.py`
- Modify: `src/api/app.py`

**Step 1: Write the failing tests**

Create `tests/api/test_completion_router.py`:

```python
import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from src.api.app import create_app
from src.api.routers.sessions import store as session_store
from src.agents.deps import AgentDependencies


def make_session(collection_name: str = "test", top_k: int = 5) -> str:
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
    application.state.model = mock_model
    yield application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_completion_session_not_found(client):
    response = await client.post(
        "/api/v1/completion",
        json={"session_id": "nonexistent", "message": "Hello"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


async def test_completion_message_too_long(client):
    session_id = make_session()
    response = await client.post(
        "/api/v1/completion",
        json={"session_id": session_id, "message": "x" * 1001},
    )
    assert response.status_code == 422


async def test_completion_returns_text_contexts_citations(client):
    session_id = make_session()

    mock_result = MagicMock()
    mock_result.data = "Docling is a document conversion library."
    mock_result.all_messages = MagicMock(return_value=[])

    async def fake_run(*args, **kwargs):
        state = session_store.get(session_id)
        state.deps.contexts = ["ctx1"]
        state.deps.citations = ["cite1"]
        return mock_result

    with patch("src.api.routers.completion.agentic_rag.run", new=fake_run):
        response = await client.post(
            "/api/v1/completion",
            json={"session_id": session_id, "message": "What is docling?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Docling is a document conversion library."
    assert data["contexts"] == ["ctx1"]
    assert data["citations"] == ["cite1"]
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/api/test_completion_router.py -v
```
Expected: Routes return 404 — completion router not registered yet

**Step 3: Implement completion router**

Create `src/api/routers/completion.py`:

```python
from loguru import logger
from fastapi import APIRouter, HTTPException, Request, status

from src.agents.agentic_rag import agentic_rag
from src.api.schemas import ChatRequest, CompletionResponse
from src.api.routers.sessions import store as session_store

router = APIRouter(prefix="/completion", tags=["completion"])


@router.post("", response_model=CompletionResponse)
async def completion(request: Request, body: ChatRequest) -> CompletionResponse:
    state = session_store.get(body.session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found",
        )

    state.deps.clear_context()
    model = request.app.state.model

    try:
        result = await agentic_rag.run(
            body.message,
            model=model,
            message_history=state.messages,
            deps=state.deps,
        )
    except Exception as e:
        logger.exception(f"Completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    state.messages.extend(result.all_messages())
    return CompletionResponse(
        text=result.data,
        contexts=state.deps.contexts or [],
        citations=state.deps.citations or [],
    )
```

**Register the router in `src/api/app.py`** — add these two lines:

```python
# add this import at top of app.py (after the chat_router import)
from src.api.routers.completion import router as completion_router

# add inside create_app(), after chat_router:
app.include_router(completion_router, prefix=settings.api_prefix)
```

**Step 4: Run tests**

```bash
uv run pytest tests/api/test_completion_router.py -v
```
Expected: 3 passed

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```
Expected: All pass

**Step 6: Commit**

```bash
git add src/api/routers/completion.py src/api/app.py tests/api/test_completion_router.py
git commit -m "feat(api): add completion endpoint"
```

---

### Task 3: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add completion endpoint to the API endpoint table**

In `README.md`, find the endpoints table and add the completion row:

```markdown
| `POST` | `/api/v1/completion` | Run agent and return full response (non-streaming) |
```

**Step 2: Add a completion example section**

After the existing "Delete a session" example, add:

```markdown
#### Get a completion (non-streaming)

```bash
curl -X POST http://localhost:8000/api/v1/completion \
  -H "Content-Type: application/json" \
  -d '{"session_id": "3f2a1b...", "message": "What is docling?"}'
```

```json
{
  "text": "Docling is a document conversion library...",
  "contexts": ["Docling is designed to..."],
  "citations": ["docling.pdf, page 3"]
}
```
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document completion endpoint in README"
```
