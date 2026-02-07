# newbieAR API Documentation

## Getting Started

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables in `.env`:
```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ADMIN_API_KEY=newbie_your_secure_admin_key_here

# Existing settings (Qdrant, LLM, etc.)
QDRANT_URI=http://localhost:6333
...
```

### Running the API

**Development:**
```bash
./scripts/run_api.sh
```

**Production:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

- **OpenAPI Docs:** http://localhost:8000/api/v1/docs
- **ReDoc:** http://localhost:8000/api/v1/redoc

## Authentication

All API requests require an API key in the header:

```bash
curl -H "X-API-Key: newbie_your_api_key" http://localhost:8000/api/v1/health
```

**Default Admin Key (Development):** `newbie_admin_dev_key_change_in_production`

## API Endpoints

### Ingestion

**Upload Document to VectorDB**
```bash
POST /api/v1/ingest/vectordb
Content-Type: multipart/form-data

file: <file>
collection_name: my_papers
chunk_strategy: hybrid
```

**Check Job Status**
```bash
GET /api/v1/ingest/jobs/{job_id}
```

**List Collections**
```bash
GET /api/v1/ingest/collections
```

### Retrieval

**Basic RAG Search**
```bash
POST /api/v1/retrieval/basic-rag
Content-Type: application/json

{
  "user_id": "user_123",
  "query": "What are transformers?",
  "collection_name": "my_papers",
  "top_k": 5
}
```

**Graph RAG Search**
```bash
POST /api/v1/retrieval/graph-rag
```

**Generate Answer**
```bash
POST /api/v1/retrieval/generate
```

### Agents

**Synchronous Chat**
```bash
POST /api/v1/agents/basic-rag/chat
Content-Type: application/json

{
  "user_id": "user_123",
  "message": "Explain transformers",
  "collection_name": "my_papers",
  "session_id": "session_xyz"
}
```

**Streaming Chat (SSE)**
```bash
POST /api/v1/agents/basic-rag/stream
```

**Session Management**
```bash
GET /api/v1/agents/sessions/{session_id}/history?user_id=user_123
POST /api/v1/agents/sessions?user_id=user_123
DELETE /api/v1/agents/sessions/{session_id}?user_id=user_123
```

## Examples

See `examples/api_client.py` for Python client examples.

## Rate Limiting

- Standard: 100 requests/minute
- Premium: 1000 requests/minute

## Error Handling

All errors return standardized JSON:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {},
    "request_id": "req_123",
    "timestamp": "2026-02-07T10:30:00Z"
  }
}
```

## Testing

```bash
pytest tests/test_api.py -v
```
