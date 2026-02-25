# FastAPI Agentic RAG — Design

**Date:** 2026-02-24
**Scope:** Core API — chat endpoint with SSE streaming + in-memory session history. No auth, no Redis, no background jobs.

---

## File Layout

```
src/api/
  __init__.py
  app.py            # FastAPI factory, lifespan, CORS
  schemas.py        # Pydantic I/O models
  session_store.py  # In-memory store: session_id → SessionState
  routers/
    __init__.py
    sessions.py     # POST /sessions, DELETE /sessions/{id}
    chat.py         # POST /chat (SSE stream)
```

Entry point:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## API Contracts

### `POST /api/v1/sessions`

Creates a session, initializing `BasicRAG` and `GraphRAG` with the given collection.

**Request:**
```json
{ "collection_name": "research_papers", "top_k": 5 }
```

**Response:**
```json
{ "session_id": "uuid4", "collection_name": "research_papers", "top_k": 5 }
```

---

### `DELETE /api/v1/sessions/{session_id}`

Destroys a session and releases its resources.

**Response:**
```json
{ "message": "Session deleted" }
```

---

### `POST /api/v1/chat`

Streams a response from the agentic RAG agent over SSE.

**Request:**
```json
{ "session_id": "uuid4", "message": "What is docling?" }
```

**Response:** `Content-Type: text/event-stream`

```
event: delta
data: {"text": "Docling is..."}

event: delta
data: {"text": " a document"}

event: done
data: {"contexts": ["..."], "citations": ["..."]}
```

On error:
```
event: error
data: {"detail": "Session not found"}
```

---

## Session State

```python
@dataclass
class SessionState:
    deps: AgentDependencies        # BasicRAG, GraphRAG, top_k, contexts, citations
    messages: list[ModelMessage]   # pydantic-ai message history for multi-turn
    collection_name: str
    top_k: int
```

Stored in a module-level `dict[str, SessionState]` in `session_store.py`. Keyed by `session_id` (UUID4).

---

## Data Flow — Chat Request

1. Look up `session_id` → `SessionState`; raise HTTP 404 if missing
2. Validate message: reject empty or > 1000 characters
3. Call `deps.clear_context()` to reset `contexts`/`citations` for the new turn
4. Call `agentic_rag.run_stream(message, model=model, message_history=state.messages, deps=state.deps)`
5. Yield `delta` SSE events for each text chunk
6. On completion: extend `state.messages` with `result.all_messages()`; yield `done` event with `contexts` + `citations`
7. On exception: yield `error` SSE event with detail string

---

## App Startup (lifespan)

The OpenAI model is initialized once at app startup via `get_openai_model()` and stored as application state. It is stateless and safe to share across requests.

---

## Constraints Preserved from CLI

- Message max length: 1000 characters
- `deps.clear_context()` called before each turn
- `state.messages` extended with `result.all_messages()` for multi-turn continuity
- Logging via loguru to `logs/agentic_rag.log`
