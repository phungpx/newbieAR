# Completion Endpoint Design

**Date:** 2026-02-25
**Status:** Approved

---

## Goal

Add `POST /api/v1/completion` — a non-streaming counterpart to `POST /api/v1/chat` that returns the full answer, contexts, and citations in a single JSON response.

---

## Architecture

A new router file `src/api/routers/completion.py` registered in `src/api/app.py` alongside the existing `sessions` and `chat` routers. The completion router shares the same `SessionStore` singleton from `sessions.py`.

---

## Request / Response

**Request** — reuses `ChatRequest` (same `session_id` + `message` with identical validation):

```python
class ChatRequest(BaseModel):
    session_id: str
    message: str  # non-empty, max 1000 chars
```

**Response** — new `CompletionResponse` schema added to `src/api/schemas.py`:

```python
class CompletionResponse(BaseModel):
    text: str
    contexts: list[str]
    citations: list[str]
```

---

## Data Flow

```
POST /api/v1/completion
  │
  ├─ look up session_id in SessionStore
  │    └─ not found → 404 HTTPException
  │
  ├─ state.deps.clear_context()
  │
  ├─ result = await agentic_rag.run(
  │      message,
  │      model=request.app.state.model,
  │      message_history=state.messages,
  │      deps=state.deps,
  │   )
  │
  ├─ state.messages.extend(result.all_messages())
  │
  └─ return CompletionResponse(
         text=result.data,
         contexts=state.deps.contexts or [],
         citations=state.deps.citations or [],
     )
```

---

## Agent Call

Uses pydantic-ai's non-streaming `Agent.run()`:

- `result.data` — the final text string
- `result.all_messages()` — full message list to extend history
- No accumulation of delta chunks needed

---

## Error Handling

| Condition | Response |
|-----------|----------|
| Session not found | `404 HTTPException` (plain JSON, consistent with REST conventions) |
| Invalid message (empty / too long) | `422 Unprocessable Entity` (FastAPI validation) |
| Agent runtime error | `500 HTTPException` with error detail |

Unlike `/chat` (which yields SSE error events), `/completion` uses standard HTTP status codes since its response is plain JSON.

---

## Files Changed

| File | Change |
|------|--------|
| `src/api/schemas.py` | Add `CompletionResponse` |
| `src/api/routers/completion.py` | New router with `POST ""` route |
| `src/api/app.py` | Register `completion_router` |
| `tests/api/test_completion_router.py` | New test file (3 tests) |

---

## Tests

| Test | Assertion |
|------|-----------|
| `test_completion_session_not_found` | 404 when session_id is unknown |
| `test_completion_message_too_long` | 422 for message > 1000 chars |
| `test_completion_returns_text_contexts_citations` | 200 with correct `text`, `contexts`, `citations` |
