# Streamlit UI Design — newbieAR Pipeline Explorer

**Date:** 2026-02-25
**Status:** Approved

## Overview

A multi-page Streamlit developer-exploration UI that drives every stage of the newbieAR pipeline (Ingestion → Retrieval → Agent → Synthesis → Evaluation) exclusively via HTTP calls to the FastAPI backend. Four new FastAPI routers are added alongside the existing chat/sessions/stream routers.

## Architecture

```
┌─────────────────────────────────────┐
│         Streamlit UI (src/ui/)       │
│                                     │
│  Home.py  (health check + nav)      │
│  api_client.py  (shared httpx)      │
│  pages/                             │
│  ├── 1_Ingestion.py                 │
│  ├── 2_Retrieval.py                 │
│  ├── 3_Agent.py                     │
│  ├── 4_Synthesis.py                 │
│  └── 5_Evaluation.py                │
└──────────────┬──────────────────────┘
               │ HTTP (FASTAPI_BASE_URL, default http://localhost:8000)
┌──────────────▼──────────────────────┐
│         FastAPI backend             │
│                                     │
│  EXISTING: /sessions /chat /chat/stream │
│                                     │
│  NEW:                               │
│    /ingest   — VectorDB + GraphDB   │
│    /retrieve — BasicRAG + GraphRAG  │
│    /synthesis — golden generation   │
│    /evaluation — deepeval metrics   │
│    /jobs/{id} — job status          │
└─────────────────────────────────────┘
```

## New FastAPI Endpoints

### `/ingest` router — `src/api/routers/ingest.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/vector` | Upload file (multipart) + `collection_name`, `chunk_strategy` → calls `VectorDBIngestion.ingest_file()` → returns `{chunks_count, collection_name}` |
| POST | `/ingest/graph` | Upload file + params → calls `GraphitiIngestion` → returns `{episodes_count}` |

### `/retrieve` router — `src/api/routers/retrieve.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/retrieve/vector` | Body: `query`, `collection_name`, `top_k`, `score_threshold?`, `rerank?` → returns list of `RetrievalInfo` |
| POST | `/retrieve/graph` | Body: `query` → returns graphiti edges/nodes |

### `/synthesis` router — `src/api/routers/synthesis.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/synthesis/jobs` | Body: `file_dir`, `output_dir`, `topic` → launches background job → returns `{job_id}` |
| GET | `/synthesis/jobs/{job_id}` | Returns `{status, goldens_count, error}` |

### `/evaluation` router — `src/api/routers/evaluation.py`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/evaluation/jobs` | Body: `goldens_dir`, `retrieval_window_size` → launches background job → returns `{job_id}` |
| GET | `/evaluation/jobs/{job_id}` | Returns `{status, metrics_summary, error}` |

### Job Store — `src/api/job_store.py`

In-memory `dict[str, JobState]` with `asyncio.Lock`. `JobState` holds:
- `status`: `pending | running | done | failed`
- `result`: stage-specific output dict
- `error`: error message string if failed

## Streamlit Pages

### `Home.py`
- Sidebar: API health indicator (`GET /health`), `FASTAPI_BASE_URL` config input
- Main: project description and links to all pages

### `1_Ingestion.py`
- Toggle: Vector DB vs Graph DB
- `st.file_uploader` (PDF)
- Params: `collection_name`, `chunk_strategy` (hybrid/hierarchical)
- Submit → `POST /ingest/vector` or `/ingest/graph`
- Result: chunk/episode count, collection name, success/error

### `2_Retrieval.py`
- Toggle: BasicRAG vs GraphRAG
- `st.text_input` for query
- Params: `collection_name`, `top_k`, `score_threshold` (float), `rerank` (bool toggle)
- Submit → `POST /retrieve/vector` or `/retrieve/graph`
- Results: expandable cards (source, score, content snippet)

### `3_Agent.py`
- Session panel: create session (`collection_name`, `top_k`), display session ID, delete button
- Chat: `st.chat_input` + `st.chat_message` history displayed in order
- Streaming via SSE (`POST /chat/stream`), token-by-token display
- Post-response: expandable contexts and citations panel

### `4_Synthesis.py`
- Params: `file_dir`, `output_dir`, `topic`
- Submit → `POST /synthesis/jobs` → store `job_id` in `st.session_state`
- Poll `GET /synthesis/jobs/{job_id}` every 3 s using `st.rerun()` + `st.spinner`
- Result: goldens count, output directory path

### `5_Evaluation.py`
- Params: `goldens_dir`, `retrieval_window_size`
- Submit → `POST /evaluation/jobs` → store `job_id`
- Poll every 3 s using `st.rerun()` + `st.spinner`
- Result: metrics table (AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall, ContextualRelevancy) — score + pass/fail per metric

## Shared Client — `src/ui/api_client.py`

```python
import httpx
import os

BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
client = httpx.Client(base_url=BASE_URL, timeout=60.0)
```

All pages import `client` from here.

## Dependencies Added

- `streamlit` — Streamlit framework
- `httpx` — promoted from test-only to main dependency (already in test extras)

## File Layout

```
src/
├── api/
│   ├── job_store.py            # NEW — in-memory job store
│   └── routers/
│       ├── ingest.py           # NEW
│       ├── retrieve.py         # NEW
│       ├── synthesis.py        # NEW
│       └── evaluation.py       # NEW
└── ui/
    ├── Home.py                 # NEW — Streamlit entry point
    ├── api_client.py           # NEW — shared httpx client
    └── pages/
        ├── 1_Ingestion.py      # NEW
        ├── 2_Retrieval.py      # NEW
        ├── 3_Agent.py          # NEW
        ├── 4_Synthesis.py      # NEW
        └── 5_Evaluation.py     # NEW
```

## Launch Commands

```bash
# Start FastAPI backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI (separate terminal)
uv run streamlit run src/ui/Home.py
```

## Constraints

- Job store is in-memory: jobs are lost on server restart. Acceptable for a developer tool.
- Synthesis and Evaluation call AWS Bedrock; they require valid AWS credentials in `.env`.
- Ingestion endpoints save files to server-local paths (`documents_dir`, `chunks_dir`); the Streamlit file uploader writes the uploaded bytes to a temp file before POSTing.
- SSE streaming in the Agent page uses `httpx` in streaming mode with `iter_lines()`.
