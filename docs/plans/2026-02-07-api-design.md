# API Design for Ingestion, Retrieval, and Agents

**Date:** 2026-02-07
**Status:** Approved
**Framework:** FastAPI
**Authentication:** API Keys
**Observability:** Langfuse Auto-Tracing

## Overview

This design covers REST APIs for the newbieAR (Newbie Agentic RAG) system, exposing three main capabilities:
1. **Ingestion API** - Async document processing into vector/graph databases
2. **Retrieval API** - Basic RAG and Graph RAG search with citations
3. **Agent API** - Agentic chat with both streaming and synchronous modes

## Architecture

### API Structure

**Base URL:** `/api/v1`

**Routers:**
- `/ingest` - Ingestion operations
- `/retrieval` - Retrieval operations
- `/agents` - Agent interactions
- `/auth` - API key management

### Authentication

**API Key Format:** `newbie_` + 32 random characters

**Header:** `X-API-Key: newbie_a8f3d9c2e1b4567890abcdef12345678`

**API Key Model:**
```python
{
  "api_key": "hashed_key",
  "user_id": "user_123",
  "name": "Production Key",
  "permissions": ["ingest", "retrieval", "agents"],
  "rate_limit_tier": "standard|premium",
  "is_active": true,
  "created_at": "timestamp",
  "last_used_at": "timestamp",
  "expires_at": "timestamp|null"
}
```

**Rate Limits:**
- Standard: 100 requests/minute
- Premium: 1000 requests/minute

## Ingestion API

### `POST /api/v1/ingest/vectordb`

**Description:** Upload and process documents into Qdrant vector database.

**Request:**
```json
{
  "file": "multipart/form-data or base64",
  "collection_name": "my_papers",
  "chunk_strategy": "hybrid|docling|semantic",
  "metadata": {"source": "research", "category": "ML"}
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "ingest_abc123",
  "status": "processing",
  "created_at": "2026-02-07T10:30:00Z",
  "message": "File queued for processing"
}
```

### `POST /api/v1/ingest/graphdb`

Similar to `/vectordb` but ingests into Neo4j/Graphiti graph database.

### `GET /api/v1/ingest/jobs/{job_id}`

**Description:** Check ingestion job status.

**Response:**
```json
{
  "job_id": "ingest_abc123",
  "status": "completed|processing|failed|queued",
  "progress": 100,
  "result": {
    "collection_name": "my_papers",
    "chunks_created": 42,
    "embeddings_added": 42,
    "file_path": "data/papers/docs/file.pdf"
  },
  "created_at": "2026-02-07T10:30:00Z",
  "completed_at": "2026-02-07T10:32:15Z",
  "error": null
}
```

### `GET /api/v1/ingest/jobs/{job_id}/result`

Returns detailed results for completed jobs.

### `GET /api/v1/ingest/collections`

Lists available collections with metadata.

**Job Management:**
- Storage: Redis (production) or in-memory (dev)
- States: `queued` → `processing` → `completed|failed`
- Timeout: 10 minutes (configurable)
- Max file size: 50MB (configurable)

## Retrieval API

### `POST /api/v1/retrieval/basic-rag`

**Description:** Semantic search using vector-only retrieval.

**Request:**
```json
{
  "user_id": "user_123",
  "query": "What are the benefits of RAG?",
  "collection_name": "my_papers",
  "top_k": 5,
  "score_threshold": 0.7,
  "return_embeddings": false
}
```

**Response:**
```json
{
  "query": "What are the benefits of RAG?",
  "results": [
    {
      "content": "RAG combines retrieval with generation...",
      "source": "paper1.pdf - Chunk #3",
      "score": 0.92,
      "metadata": {"chunk_id": 3, "filename": "paper1.pdf"}
    }
  ],
  "citations": [
    {
      "citation_id": 1,
      "source": "paper1.pdf - Chunk #3",
      "content_snippet": "RAG combines retrieval with generation...",
      "relevance_score": 0.92,
      "cited_in_answer": false
    }
  ],
  "retrieval_time_ms": 145,
  "num_results": 5
}
```

### `POST /api/v1/retrieval/graph-rag`

**Description:** Hybrid search combining vector search and graph traversal.

**Request (extends basic-rag):**
```json
{
  "user_id": "user_123",
  "query": "How do transformers work?",
  "collection_name": "my_papers",
  "top_k": 5,
  "graph_depth": 2,
  "enable_reranking": true
}
```

**Response:**
```json
{
  "query": "How do transformers work?",
  "results": [...],
  "graph_paths": [
    {
      "entities": ["Transformer", "Attention", "BERT"],
      "relationship": "uses",
      "evidence_chunks": [5, 12]
    }
  ],
  "citations": [
    {
      "citation_id": 1,
      "source": "attention_paper.pdf - Chunk #5",
      "content_snippet": "Transformers use self-attention...",
      "relevance_score": 0.94,
      "cited_in_answer": true
    }
  ],
  "retrieval_time_ms": 280
}
```

### `POST /api/v1/retrieval/generate`

**Description:** Retrieve + generate answer (non-agentic, direct LLM call).

Combines retrieval with LLM generation. Response includes both `results` and `generated_answer` fields. Citations marked with `cited_in_answer: true` if used in generation.

**Validation:**
- Query: 1-1000 characters
- top_k: 1-100
- Collection must exist (404 if not)

## Agent API

### `POST /api/v1/agents/basic-rag/chat` (Synchronous)

**Description:** Agentic conversation with BasicRAG tool access, returns full response.

**Request:**
```json
{
  "user_id": "user_123",
  "message": "What are transformers?",
  "collection_name": "my_papers",
  "session_id": "session_xyz",
  "top_k": 5,
  "include_history": true
}
```

**Response:**
```json
{
  "message": "Transformers are neural network architectures...",
  "tool_calls": [
    {
      "tool": "search_basic_rag",
      "query": "transformers architecture",
      "results_count": 5,
      "token_usage": {
        "embedding_tokens": 8,
        "prompt_tokens": 0,
        "completion_tokens": 0
      },
      "execution_time_ms": 145
    }
  ],
  "citations": [
    {
      "citation_id": 1,
      "source": "attention_paper.pdf - Chunk #8",
      "content_snippet": "The Transformer model architecture...",
      "relevance_score": 0.95,
      "cited_in_answer": true
    }
  ],
  "session_id": "session_xyz",
  "user_id": "user_123",
  "token_usage": {
    "total_prompt_tokens": 1240,
    "total_completion_tokens": 320,
    "total_tokens": 1560,
    "breakdown": {
      "retrieval_embedding_tokens": 8,
      "llm_prompt_tokens": 1240,
      "llm_completion_tokens": 320
    }
  },
  "response_time_ms": 2300
}
```

### `POST /api/v1/agents/basic-rag/stream` (SSE Streaming)

**Description:** Same as `/chat` but streams response via Server-Sent Events.

**Request:** Same as `/chat`

**Response (SSE Stream):**
```
event: token
data: {"delta": "Transformers"}

event: token
data: {"delta": " are"}

event: tool_call
data: {"tool": "search_basic_rag", "status": "executing"}

event: tool_call
data: {"tool": "search_basic_rag", "status": "completed", "token_usage": {"embedding_tokens": 8}}

event: citations
data: [{"citation_id": 1, "source": "...", "relevance_score": 0.95}]

event: done
data: {"token_usage": {"total_tokens": 1560, "breakdown": {...}}, "response_time_ms": 2300}
```

### `POST /api/v1/agents/graph-rag/chat`
### `POST /api/v1/agents/graph-rag/stream`

Same structure as basic-rag endpoints but uses GraphRAG tool with graph traversal capabilities.

### Session Management

**`GET /api/v1/agents/sessions/{session_id}/history`**

**Query params:** `user_id` (required), `limit` (default: 50), `offset`

**Response:**
```json
{
  "session_id": "session_xyz",
  "user_id": "user_123",
  "messages": [
    {
      "role": "user",
      "content": "What are transformers?",
      "timestamp": "2026-02-07T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Transformers are...",
      "tool_calls": [...],
      "citations": [...],
      "token_usage": {...},
      "timestamp": "2026-02-07T10:30:03Z"
    }
  ],
  "total_messages": 2,
  "created_at": "2026-02-07T10:30:00Z"
}
```

**Additional Session Endpoints:**
- `POST /api/v1/agents/sessions` - Create new session
- `DELETE /api/v1/agents/sessions/{session_id}` - Clear history
- `GET /api/v1/agents/sessions?user_id=user_123` - List user's sessions

**Storage:**
- Redis (production) or in-memory (dev)
- TTL: 24 hours (configurable)
- Auto-create if `session_id` not provided
- User isolation via `user_id` validation

## Data Models

### Request Models

```python
# Ingestion
class IngestRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=100)
    chunk_strategy: ChunkStrategy = ChunkStrategy.HYBRID
    metadata: dict[str, Any] = Field(default_factory=dict)

# Retrieval
class RetrievalRequest(BaseModel):
    user_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class GraphRAGRequest(RetrievalRequest):
    graph_depth: int = Field(default=2, ge=1, le=5)
    enable_reranking: bool = True

# Agent
class AgentChatRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    collection_name: str
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_history: bool = True
```

### Response Models

```python
class Citation(BaseModel):
    citation_id: int
    source: str
    content_snippet: str
    relevance_score: float
    cited_in_answer: bool = False

class TokenUsage(BaseModel):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    breakdown: dict[str, int]

class ToolCallInfo(BaseModel):
    tool: str
    query: str | None = None
    results_count: int = 0
    token_usage: dict[str, int]
    execution_time_ms: int

class AgentChatResponse(BaseModel):
    message: str
    tool_calls: list[ToolCallInfo]
    citations: list[Citation]
    session_id: str
    user_id: str
    token_usage: TokenUsage
    response_time_ms: int
```

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "INVALID_COLLECTION",
    "message": "Collection 'my_papers' does not exist",
    "details": {
      "collection_name": "my_papers",
      "available_collections": ["papers_2024", "research_notes"]
    },
    "request_id": "req_abc123",
    "timestamp": "2026-02-07T10:30:00Z"
  }
}
```

### Error Codes

| HTTP | Error Code | Description |
|------|------------|-------------|
| 400 | INVALID_REQUEST | Validation error, malformed input |
| 401 | INVALID_API_KEY | Missing or invalid API key |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 413 | FILE_TOO_LARGE | Upload exceeds size limit |
| 422 | VALIDATION_ERROR | Pydantic validation failed |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests (includes Retry-After header) |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Database connection failed |

## Observability

### Langfuse Auto-Tracing

**Middleware captures:**
1. **API Requests** - Endpoint, method, user_id, payloads, status, latency
2. **LLM Calls** - Model, prompt, completion, token usage, cost
3. **Retrieval Operations** - Query, embedding time, search latency, results
4. **Tool Executions** - Tool name, parameters, execution time, status

**Dashboard Metrics:**
- Request volume by endpoint
- P50/P95/P99 latency
- Token usage & cost by user
- Error rates and types
- Session conversation flows
- Citation usage patterns

**Configuration:**
```python
langfuse_enabled: bool = True
langfuse_public_key: str
langfuse_secret_key: str
langfuse_host: str = "https://cloud.langfuse.com"
```

## Project Structure

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, middleware, startup
│   ├── dependencies.py            # Auth, rate limiting dependencies
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Ingestion endpoints
│   │   ├── retrieval.py          # Retrieval endpoints
│   │   ├── agents.py             # Agent endpoints
│   │   └── auth.py               # API key management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py           # Request Pydantic models
│   │   ├── responses.py          # Response Pydantic models
│   │   └── api_key.py            # API key storage model
│   ├── services/
│   │   ├── __init__.py
│   │   ├── job_manager.py        # Background job tracking
│   │   ├── session_manager.py   # Conversation history
│   │   └── auth_service.py      # API key validation
│   └── middleware/
│       ├── __init__.py
│       ├── langfuse.py           # Observability middleware
│       ├── rate_limit.py         # Rate limiting
│       └── error_handler.py     # Global exception handling
├── ingestion/                     # Existing code
├── retrieval/                     # Existing code
├── agents/                        # Existing code
├── models/                        # Existing shared models
├── settings.py                    # Extended with API settings
└── deps/                          # Existing dependencies
```

## Configuration Extensions

```python
# settings.py additions

# API Settings
api_host: str = "0.0.0.0"
api_port: int = 8000
api_prefix: str = "/api/v1"
cors_origins: list[str] = ["http://localhost:3000"]

# Auth
api_keys_storage: str = "redis"  # or "sqlite", "postgresql"
admin_api_key: str  # for key management

# Rate Limiting
rate_limit_enabled: bool = True
rate_limit_standard: str = "100/minute"
rate_limit_premium: str = "1000/minute"

# Jobs
job_storage: str = "redis"  # or "memory"
job_timeout_seconds: int = 600
max_file_size_mb: int = 50

# Sessions
session_storage: str = "redis"  # or "memory"
session_ttl_hours: int = 24
max_history_messages: int = 20
```

## Deployment

### Production Stack

**Server:** Uvicorn + Gunicorn (async workers)
**Reverse Proxy:** Nginx (SSL, load balancing)
**Storage:** Redis (sessions/jobs), PostgreSQL (API keys)
**Monitoring:** Langfuse + Prometheus + health checks

### Docker Compose

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:5432/newbiear
    depends_on: [redis, postgres, qdrant]

  redis:
    image: redis:7-alpine
    volumes: ["redis_data:/data"]

  postgres:
    image: postgres:16-alpine
    volumes: ["postgres_data:/var/lib/postgresql/data"]

  qdrant:
    image: qdrant/qdrant:latest
    volumes: ["qdrant_data:/qdrant/storage"]
```

### Health Checks

- `GET /health` - Basic liveness
- `GET /health/ready` - Readiness (DB connections)
- `GET /health/db` - Vector/Graph DB status

### Performance Targets

- Ingestion: 50MB files in <5 minutes
- Retrieval: P95 latency <200ms
- Agents (sync): P95 latency <3s
- Streaming: First token <500ms

## Implementation Phases

### Phase 1: Core APIs (Week 1)
- FastAPI setup, router structure
- Authentication middleware
- Ingestion & retrieval endpoints
- Basic error handling

### Phase 2: Agent APIs (Week 2)
- Synchronous agent endpoints
- SSE streaming implementation
- Session management
- Token tracking

### Phase 3: Observability (Week 3)
- Langfuse middleware
- Rate limiting
- Health checks & metrics
- API key management endpoints

### Phase 4: Testing & Docs (Week 4)
- Unit tests (pytest)
- Integration tests
- OpenAPI docs refinement
- Deployment setup

## Testing Strategy

**Unit Tests:**
- Request/response model validation
- Auth service logic
- Job manager state transitions
- Session manager operations

**Integration Tests:**
- End-to-end API flows
- Database interactions
- Background job processing
- Streaming responses

**Load Tests:**
- Rate limiting effectiveness
- Concurrent request handling
- Token tracking accuracy
- Session storage performance

## Security Considerations

1. **API Keys:** Hashed storage (bcrypt/argon2), HTTPS required
2. **CORS:** Configured allowed origins
3. **Input Validation:** Pydantic models, size limits
4. **SQL Injection:** Prevented via ORM/parameterized queries
5. **Rate Limiting:** Per-user enforcement
6. **Session Isolation:** User-scoped access control

## Future Enhancements

- WebSocket support for bidirectional agent communication
- Batch ingestion API for multiple files
- Advanced reranking with cross-encoders
- Caching layer for frequent queries
- Multi-tenancy with organization-level isolation
- GraphQL API alongside REST
- Export conversation history to various formats
