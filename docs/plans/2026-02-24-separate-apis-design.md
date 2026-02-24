# Separate APIs Design: Ingestion, Retrieval, Agents

**Date:** 2026-02-24
**Branch:** `features/add_ui`

## Summary

Add three domain-specific REST API routers to the FastAPI application, exposing the existing `VectorDBIngestion`, `GraphitiIngestion`, `BasicRAG`, `GraphRetrieval`, `agentic_basic_rag`, and `agentic_graph_rag` business logic over HTTP.

## Decisions

- **Style:** Plain REST (not OpenAI-compatible)
- **Ingestion input:** File upload via multipart/form-data
- **Ingestion execution:** Async with job polling
- **Agent sessions:** Stateful, Redis-backed session history
- **Structure:** Single FastAPI app, domain-organized

## Directory Structure

```
src/api/
  app.py                      # FastAPI lifespan, router registration
  deps.py                     # Shared FastAPI dependencies
  ingestion/
    __init__.py
    router.py                 # POST /api/v1/ingestion/vectordb, /graphdb
    schemas.py                # IngestionJobResponse, JobStatusResponse
    service.py                # IngestionService
    job_store.py              # In-memory or Redis job store
  retrieval/
    __init__.py
    router.py                 # POST /api/v1/retrieval/basic, /graph
    schemas.py                # RetrievalRequest, RetrievalResponse
    service.py                # RetrievalService
  agents/
    __init__.py
    router.py                 # POST /api/v1/agents/basic, /graph (SSE)
    schemas.py                # AgentRequest, AgentResponse
    service.py                # AgentService
    session_store.py          # Redis-backed session store
```

## Endpoints

### Ingestion (`/api/v1/ingestion`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/ingestion/vectordb` | Upload PDF → Qdrant; returns `job_id` (202) |
| POST | `/api/v1/ingestion/graphdb` | Upload PDF → Neo4j/Graphiti; returns `job_id` (202) |
| GET | `/api/v1/ingestion/jobs/{job_id}` | Poll job status |

**Request (multipart/form-data):**
```
file: <binary>
collection_name: str        # vectordb only
chunk_strategy: hybrid|hierarchical
```

**Job response:**
```json
{ "job_id": "abc123", "status": "pending|running|done|failed", "result": {...} }
```

### Retrieval (`/api/v1/retrieval`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/retrieval/basic` | Vector search against Qdrant |
| POST | `/api/v1/retrieval/graph` | Graph search via Graphiti/Neo4j |

**Basic request:**
```json
{ "query": "...", "collection_name": "research_papers", "top_k": 5 }
```

**Basic response:**
```json
{ "results": [{ "content": "...", "source": "...", "score": 0.92 }] }
```

**Graph request:**
```json
{ "query": "...", "top_k": 5 }
```

**Graph response:**
```json
{ "nodes": [...], "edges": [...], "episodes": [...] }
```

### Agents (`/api/v1/agents`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/agents/basic` | Agentic BasicRAG with SSE streaming |
| POST | `/api/v1/agents/graph` | Agentic GraphRAG with SSE streaming |
| DELETE | `/api/v1/agents/sessions/{session_id}` | Clear session history |

**Request:**
```json
{ "query": "...", "session_id": "optional", "collection_name": "research_papers", "top_k": 5 }
```

**SSE stream:**
```
data: {"type": "token", "content": "Hello"}
data: {"type": "token", "content": " world"}
data: {"type": "done", "session_id": "xyz", "citations": ["doc.pdf - Chunk #1"]}
```

## Data Flow

### Ingestion
1. Client POSTs multipart file + params
2. API saves file to temp path, creates job record (`pending`), spawns `asyncio.create_task`
3. Background task runs ingestion class, updates job to `running` → `done`/`failed`
4. Client polls `GET /jobs/{job_id}` for result

### Retrieval
1. Client POSTs query params
2. Service calls `.retrieve()` synchronously (no streaming)
3. Returns list of results immediately

### Agents
1. Client POSTs query + optional `session_id`
2. Service loads message history from session store (creates new session if absent)
3. Runs pydantic-ai `agent.run_stream()`, yields SSE tokens
4. On completion: saves history to session store, emits final `done` event with `session_id` + citations

## Storage

Controlled by existing `settings.py`:
- `JobSettings.job_storage`: `"memory"` (dict) or `"redis"`
- `SessionSettings.session_storage`: `"memory"` or `"redis"`
- `SessionSettings.session_ttl_hours`: default 24h
- `JobSettings.job_timeout_seconds`: default 600s

## Error Handling

- Ingestion job failures: stored in job record with error message, status `"failed"`
- Retrieval errors: standard FastAPI 500 with detail
- Agent errors: SSE event `{"type": "error", "message": "..."}` before stream close
- Invalid session_id: create new session, return new `session_id`

## Files to Create

```
src/api/app.py
src/api/deps.py
src/api/ingestion/__init__.py
src/api/ingestion/router.py
src/api/ingestion/schemas.py
src/api/ingestion/service.py
src/api/ingestion/job_store.py
src/api/retrieval/__init__.py
src/api/retrieval/router.py
src/api/retrieval/schemas.py
src/api/retrieval/service.py
src/api/agents/__init__.py
src/api/agents/router.py
src/api/agents/schemas.py
src/api/agents/service.py
src/api/agents/session_store.py
tests/api/test_ingestion.py
tests/api/test_retrieval.py
tests/api/test_agents.py
```
