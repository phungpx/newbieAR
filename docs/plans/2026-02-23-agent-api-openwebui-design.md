# Agent API + Open WebUI Design

**Date:** 2026-02-23
**Status:** Approved

## Goal

Expose `basic_rag_agent` (Qdrant vector search) and `graphiti_agent` (Neo4j/Graphiti graph search) as an OpenAI-compatible HTTP API, with Open WebUI as the browser UI. Both agents appear as selectable "models" in Open WebUI.

---

## Section 1: API Layer Structure

New `src/api/` package, entry point `src/api/app.py`. Existing `main.py` is untouched.

```
src/api/
├── app.py              # FastAPI app + lifespan (Redis connection pool)
├── routers/
│   ├── models.py       # GET /v1/models
│   └── chat.py         # POST /v1/chat/completions (streaming SSE)
├── services/
│   ├── session.py      # Redis session store: load/save pydantic-ai ModelMessage lists
│   └── agent_runner.py # Dispatch to basic_rag_agent or graphiti_agent, yield SSE deltas
└── schemas.py          # OpenAI-compatible Pydantic request/response models
```

### Models exposed

| Model ID    | Agent            | Backend              |
|-------------|------------------|----------------------|
| `basic-rag` | `basic_rag_agent` | Qdrant (collection from `settings.qdrant_collection_name`) |
| `graph-rag` | `graphiti_agent`  | Neo4j / Graphiti     |

### LLM model routing

The `model` field in the chat request also drives LLM selection:
- Starts with `gemini-` → `GoogleModel` (Vertex AI, project from `agentic_*` files)
- Anything else → `OpenAIChatModel` (using `settings.llm_base_url` + `settings.llm_api_key`)

`model` field is independent of the agent selection — the agent is chosen by a separate `X-Agent-Type` header or, if absent, defaults based on the model id prefix (`basic-rag` / `graph-rag`).

**Design decision:** `model` in the request body carries both the agent selector and LLM selector. Naming convention:
- `basic-rag` → basic_rag_agent + OpenAI-compatible LLM from settings
- `graph-rag` → graphiti_agent + OpenAI-compatible LLM from settings
- `basic-rag/gemini-2.5-flash` → basic_rag_agent + Google Gemini
- `graph-rag/gemini-2.5-flash` → graphiti_agent + Google Gemini

`/v1/models` enumerates all supported combinations.

---

## Section 2: Session Management (Redis)

**Session key derivation:**
SHA-256 of `(model + first_user_message_content)` → stable key for the conversation.
Alternatively, clients may pass `X-Session-Id: <uuid>` header to override.

**Storage schema:**
```
redis key:  session:<sha256>
value:      JSON array of serialized pydantic-ai ModelMessage objects
TTL:        settings.session_ttl_hours (default 24h)
```

**Per-request flow:**
1. Derive/read session key from request
2. Load `prior_messages` from Redis (empty list if new session)
3. Run agent: `agent.run_stream(user_input, message_history=prior_messages, deps=deps)`
4. Stream SSE deltas to client
5. After stream completes, append `result.all_messages()` back to Redis

**Redis connection:** `redis.asyncio.Redis` pool created in FastAPI lifespan, injected via dependency.

---

## Section 3: Streaming & OpenAI Wire Format

**Endpoint:** `POST /v1/chat/completions`
**Response:** `StreamingResponse(media_type="text/event-stream")`

Each delta chunk:
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"<model>","choices":[{"index":0,"delta":{"content":"<text>"},"finish_reason":null}]}
```

Final chunk:
```
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","model":"<model>","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Source of deltas:** `agent.run_stream(...).stream_text(delta=True)` — each yielded string becomes one SSE chunk. No buffering.

**Non-streaming fallback:** If `stream=false` in request, collect all deltas and return a standard `chat.completion` object.

---

## Section 4: Docker & Open WebUI

**File:** `infras/docker-compose.openwebui.yaml`

Three services on shared network `newbie-ar-net`:

### `newbie-ar-api`
- Build: `Dockerfile.api` at repo root
- Ports: `8000:8000`
- Env: mounts `.env` file
- Depends on: `redis`

### `redis`
- Image: `redis:7-alpine`
- Ports: `6379:6379` (local dev only)
- Volume: `redis_data`

### `open-webui`
- Image: `ghcr.io/open-webui/open-webui:main`
- Ports: `3000:8080`
- Key env vars:
  ```
  OPENAI_API_BASE_URL=http://newbie-ar-api:8000/v1
  OPENAI_API_KEY=empty
  WEBUI_AUTH=false
  ```
- Volume: `openwebui_data`

### `Dockerfile.api`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev
COPY src/ src/
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Out of Scope

- Authentication on the API (Open WebUI handles auth internally)
- Ingestion endpoints (existing separate pipeline)
- Evaluation/synthesis endpoints
