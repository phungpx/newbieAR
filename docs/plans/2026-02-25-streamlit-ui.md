# Streamlit UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-page Streamlit developer-exploration UI that drives every stage of the newbieAR pipeline (Ingestion → Retrieval → Agent → Synthesis → Evaluation) via HTTP calls to the FastAPI backend.

**Architecture:** Four new FastAPI routers (`/ingest`, `/retrieve`, `/synthesis`, `/evaluation`) plus an in-memory job store for long-running background tasks. A Streamlit multi-page app under `src/ui/` calls all endpoints through a shared `httpx` client.

**Tech Stack:** Streamlit, httpx, FastAPI BackgroundTasks, pydantic, pytest + httpx AsyncClient

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add streamlit and promote httpx to main deps**

In `pyproject.toml`, add to `dependencies`:
```toml
"streamlit>=1.40.0",
"httpx>=0.28",
```

Remove `"httpx>=0.28"` from `[project.optional-dependencies] test` (it moves to main).

**Step 2: Install**

```bash
uv sync
```

Expected: resolves without error.

**Step 3: Verify**

```bash
uv run python -c "import streamlit; import httpx; print('ok')"
```

Expected: `ok`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add streamlit and httpx as main dependencies"
```

---

## Task 2: Create in-memory job store

**Files:**
- Create: `src/api/job_store.py`
- Create: `tests/api/test_job_store.py`

**Step 1: Write failing tests**

Create `tests/api/test_job_store.py`:

```python
import pytest
from src.api.job_store import JobStore, JobStatus


def test_create_job_returns_unique_id():
    store = JobStore()
    id1 = store.create()
    id2 = store.create()
    assert id1 != id2


def test_new_job_is_pending():
    store = JobStore()
    job_id = store.create()
    job = store.get(job_id)
    assert job is not None
    assert job["status"] == JobStatus.PENDING


def test_update_job_status():
    store = JobStore()
    job_id = store.create()
    store.update(job_id, status=JobStatus.RUNNING)
    assert store.get(job_id)["status"] == JobStatus.RUNNING


def test_complete_job_with_result():
    store = JobStore()
    job_id = store.create()
    store.update(job_id, status=JobStatus.DONE, result={"count": 3})
    job = store.get(job_id)
    assert job["status"] == JobStatus.DONE
    assert job["result"]["count"] == 3


def test_fail_job_with_error():
    store = JobStore()
    job_id = store.create()
    store.update(job_id, status=JobStatus.FAILED, error="something went wrong")
    job = store.get(job_id)
    assert job["status"] == JobStatus.FAILED
    assert job["error"] == "something went wrong"


def test_get_unknown_job_returns_none():
    store = JobStore()
    assert store.get("nonexistent") is None
```

**Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/api/test_job_store.py -v
```

Expected: `ImportError` — `job_store` module not found.

**Step 3: Implement `src/api/job_store.py`**

```python
import uuid
from enum import str, auto
from typing import Any


class JobStatus(str):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
        }
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        if job_id not in self._jobs:
            return
        self._jobs[job_id]["status"] = status
        if result is not None:
            self._jobs[job_id]["result"] = result
        if error is not None:
            self._jobs[job_id]["error"] = error
```

**Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/api/test_job_store.py -v
```

Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add src/api/job_store.py tests/api/test_job_store.py
git commit -m "feat: add in-memory JobStore for background task tracking"
```

---

## Task 3: Ingestion router

**Files:**
- Create: `src/api/routers/ingest.py`
- Create: `tests/api/test_ingest_router.py`

**Context:**
- `VectorDBIngestion(documents_dir, chunks_dir, qdrant_collection_name, chunk_strategy).ingest_file(file_path)` — synchronous, returns `{file_save_path, chunk_save_path, qdrant_collection_name}`
- `GraphitiIngestion(output_dir, chunk_strategy).ingest_file(file_path)` — async (needs `await`)
- Ingestion accepts a file upload; we write it to a temp file on disk before calling ingest.

**Step 1: Write failing tests**

Create `tests/api/test_ingest_router.py`:

```python
import io
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from src.api.app import app


@pytest.fixture
def pdf_bytes():
    return b"%PDF-1.4 fake pdf content"


@pytest.mark.asyncio
async def test_ingest_vector_returns_200(pdf_bytes):
    mock_result = {
        "file_save_path": "/tmp/doc.md",
        "chunk_save_path": "/tmp/chunks.json",
        "qdrant_collection_name": "test_col",
    }
    with patch(
        "src.api.routers.ingest.VectorDBIngestion"
    ) as MockIngest:
        instance = MockIngest.return_value
        instance.ingest_file.return_value = mock_result

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingest/vector",
                data={"collection_name": "test_col", "chunk_strategy": "hybrid"},
                files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "test_col"
    assert "chunks_count" in body


@pytest.mark.asyncio
async def test_ingest_graph_returns_200(pdf_bytes):
    with patch(
        "src.api.routers.ingest.GraphitiIngestion"
    ) as MockIngest:
        instance = MockIngest.return_value
        instance.ingest_file = AsyncMock(return_value=None)
        instance.close = AsyncMock(return_value=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingest/graph",
                files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

**Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/api/test_ingest_router.py -v
```

Expected: FAIL — router not registered.

**Step 3: Implement `src/api/routers/ingest.py`**

```python
import tempfile
import asyncio
from pathlib import Path
from loguru import logger
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status

from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.ingestion.ingest_graphdb import GraphitiIngestion
from src.models import ChunkStrategy

router = APIRouter(prefix="/ingest", tags=["ingest"])

_TEMP_DOCS_DIR = "data/api/docs"
_TEMP_CHUNKS_DIR = "data/api/chunks"


@router.post("/vector")
async def ingest_vector(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_strategy: str = Form(ChunkStrategy.HYBRID.value),
):
    if chunk_strategy not in [e.value for e in ChunkStrategy]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid chunk_strategy. Choose from: {[e.value for e in ChunkStrategy]}",
        )

    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename).suffix, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline = VectorDBIngestion(
            documents_dir=_TEMP_DOCS_DIR,
            chunks_dir=_TEMP_CHUNKS_DIR,
            qdrant_collection_name=collection_name,
            chunk_strategy=chunk_strategy,
        )
        result = await asyncio.to_thread(pipeline.ingest_file, tmp_path)
        logger.info(f"Vector ingest done: {result}")
    except Exception as e:
        logger.exception(f"Vector ingest failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    return {
        "status": "ok",
        "collection_name": result["qdrant_collection_name"],
        "chunks_count": 0,  # VectorDBIngestion doesn't return chunk count; extend if needed
        "chunk_save_path": result["chunk_save_path"],
    }


@router.post("/graph")
async def ingest_graph(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(ChunkStrategy.HIERARCHICAL.value),
):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename).suffix, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline = GraphitiIngestion(chunk_strategy=chunk_strategy)
        await pipeline.ingest_file(tmp_path)
        await pipeline.close()
        logger.info(f"Graph ingest done for {file.filename}")
    except Exception as e:
        logger.exception(f"Graph ingest failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    return {"status": "ok", "filename": file.filename}
```

**Step 4: Register router in `src/api/app.py`**

Add after existing router imports:
```python
from src.api.routers.ingest import router as ingest_router
```

Add inside `create_app()` after existing `include_router` calls:
```python
app.include_router(ingest_router, prefix=settings.api_prefix)
```

**Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/api/test_ingest_router.py -v
```

Expected: 2 tests PASS.

**Step 6: Commit**

```bash
git add src/api/routers/ingest.py tests/api/test_ingest_router.py src/api/app.py
git commit -m "feat: add /ingest router for vector and graph ingestion"
```

---

## Task 4: Retrieval router

**Files:**
- Create: `src/api/routers/retrieve.py`
- Create: `tests/api/test_retrieve_router.py`

**Context:**
- `BasicRAG(qdrant_collection_name).retrieve(query, top_k, score_threshold)` → `list[RetrievalInfo]` (each has `content`, `source`, `score`)
- `GraphRAG().retrieve(query, top_k)` → `(contexts: list[str], citations: list[str])`
- Both are async.

**Step 1: Write failing tests**

Create `tests/api/test_retrieve_router.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from src.api.app import app
from src.models import RetrievalInfo


@pytest.mark.asyncio
async def test_retrieve_vector_returns_results():
    mock_results = [
        RetrievalInfo(content="Doc content", source="file.pdf - Chunk #1", score=0.92)
    ]
    with patch("src.api.routers.retrieve.BasicRAG") as MockRAG:
        instance = MockRAG.return_value
        instance.retrieve = AsyncMock(return_value=mock_results)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieve/vector",
                json={
                    "query": "what is docling?",
                    "collection_name": "research_papers",
                    "top_k": 5,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["score"] == 0.92


@pytest.mark.asyncio
async def test_retrieve_graph_returns_results():
    with patch("src.api.routers.retrieve.GraphRAG") as MockRAG:
        instance = MockRAG.return_value
        instance.initialize_graphiti_client = AsyncMock()
        instance.retrieve = AsyncMock(
            return_value=(["Node: Docling is a tool"], ["file-docling-chunk-0"])
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieve/graph",
                json={"query": "what is docling?", "top_k": 5},
            )

    assert response.status_code == 200
    body = response.json()
    assert len(body["contexts"]) == 1
    assert len(body["citations"]) == 1
```

**Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/api/test_retrieve_router.py -v
```

Expected: FAIL — router not found.

**Step 3: Implement `src/api/routers/retrieve.py`**

```python
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG


class VectorRetrieveRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = False


class GraphRetrieveRequest(BaseModel):
    query: str
    top_k: int = 10


router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("/vector")
async def retrieve_vector(body: VectorRetrieveRequest):
    try:
        rag = BasicRAG(qdrant_collection_name=body.collection_name)
        if body.rerank:
            from src.deps.cross_encoder.sentence_transformers_reranker import (
                SentenceTransformersCrossEncoder,
            )
            rag.cross_encoder = SentenceTransformersCrossEncoder()
        results = await rag.retrieve(
            query=body.query,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
        return {"results": [r.model_dump() for r in results]}
    except Exception as e:
        logger.exception(f"Vector retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/graph")
async def retrieve_graph(body: GraphRetrieveRequest):
    try:
        rag = GraphRAG()
        contexts, citations = await rag.retrieve(query=body.query, top_k=body.top_k)
        await rag.close()
        return {"contexts": contexts, "citations": citations}
    except Exception as e:
        logger.exception(f"Graph retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
```

**Step 4: Register router in `src/api/app.py`**

Add import:
```python
from src.api.routers.retrieve import router as retrieve_router
```

Add in `create_app()`:
```python
app.include_router(retrieve_router, prefix=settings.api_prefix)
```

**Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/api/test_retrieve_router.py -v
```

Expected: 2 tests PASS.

**Step 6: Commit**

```bash
git add src/api/routers/retrieve.py tests/api/test_retrieve_router.py src/api/app.py
git commit -m "feat: add /retrieve router for vector and graph search"
```

---

## Task 5: Synthesis router

**Files:**
- Create: `src/api/routers/synthesis.py`

**Context:**
- Long-running job: `generate_contexts()` (async) + `synthesizer.generate_goldens_from_contexts()` (sync) + `save_goldens_to_files()`
- FastAPI `BackgroundTasks` runs the task in a thread; we use `asyncio.run()` inside for the async portion.
- Use the module-level `JobStore` (singleton shared across the app). Import from `src.api.job_store`.
- The synthesis endpoint imports from `src.synthesis.synthesize` and `src.synthesis.generate_contexts` instead of re-creating the Synthesizer (avoid duplicating config).

> **Warning:** Importing `src.synthesis.synthesize` creates module-level objects (GPTModel, Synthesizer, embedder, vector_store) at import time. This is acceptable for a dev tool — it just means FastAPI startup is slightly slower and requires valid env vars.

**Step 1: Implement `src/api/routers/synthesis.py`**

```python
import asyncio
from pathlib import Path
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.api.job_store import JobStore, JobStatus
from src.synthesis.synthesize import (
    synthesizer,
    embedder,
    vector_store,
    model,
    STYLING_CONFIG,
    Topic,
)
from src.synthesis.generate_contexts import generate_contexts, save_goldens_to_files
from src.settings import settings

router = APIRouter(prefix="/synthesis", tags=["synthesis"])
job_store = JobStore()


class SynthesisRequest(BaseModel):
    file_dir: str
    output_dir: str = "data/goldens"
    topic: str = Topic.RESEARCH_PAPER.value
    num_contexts: int = 5
    context_size: int = 5


def _run_synthesis(job_id: str, request: SynthesisRequest) -> None:
    job_store.update(job_id, status=JobStatus.RUNNING)
    try:
        file_dir = Path(request.file_dir)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = list(file_dir.glob("**/*.*"))
        total_goldens = 0

        for file_path in file_paths:
            logger.info(f"Synthesizing {file_path}")
            contexts = asyncio.run(
                generate_contexts(
                    str(file_path),
                    model=model,
                    embedder=embedder,
                    vector_store=vector_store,
                    embedding_size=settings.embedding_dimensions,
                    num_contexts=request.num_contexts,
                    context_size=request.context_size,
                )
            )
            if not contexts:
                continue
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
                source_files=[str(file_path)] * len(contexts),
            )
            save_goldens_to_files(goldens, output_dir)
            total_goldens += len(goldens)

        job_store.update(
            job_id,
            status=JobStatus.DONE,
            result={"goldens_count": total_goldens, "output_dir": str(output_dir)},
        )
    except Exception as e:
        logger.exception(f"Synthesis job {job_id} failed: {e}")
        job_store.update(job_id, status=JobStatus.FAILED, error=str(e))


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_synthesis_job(
    body: SynthesisRequest, background_tasks: BackgroundTasks
):
    if not Path(body.file_dir).exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"file_dir does not exist: {body.file_dir}",
        )
    job_id = job_store.create()
    background_tasks.add_task(_run_synthesis, job_id, body)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_synthesis_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )
    return job
```

**Step 2: Register router in `src/api/app.py`**

Add import:
```python
from src.api.routers.synthesis import router as synthesis_router
```

Add in `create_app()`:
```python
app.include_router(synthesis_router, prefix=settings.api_prefix)
```

**Step 3: Manual smoke test**

```bash
# Start FastAPI
uvicorn src.api.app:app --port 8000 --reload

# In another terminal
curl -X POST http://localhost:8000/api/v1/synthesis/jobs \
  -H "Content-Type: application/json" \
  -d '{"file_dir": "data/papers/files", "output_dir": "data/goldens", "topic": "paper"}'
# Expected: {"job_id": "<uuid>"}

curl http://localhost:8000/api/v1/synthesis/jobs/<job_id>
# Expected: {"status": "pending"|"running"|"done"|"failed", ...}
```

**Step 4: Commit**

```bash
git add src/api/routers/synthesis.py src/api/app.py
git commit -m "feat: add /synthesis router with background job support"
```

---

## Task 6: Evaluation router

**Files:**
- Create: `src/api/routers/evaluation.py`

**Context:**
- `create_metrics()` — creates deepeval metric wrappers (uses Bedrock critique model)
- `create_llm_test_case(file_path, retrieval_window_size, collection_name)` — calls `BasicRAG().generate()` which is **async**; wraps with `asyncio.run()`
- `evaluate_llm_test_case_on_metrics(test_case, metrics)` — synchronous
- Job loops over all `*.json` files in `goldens_dir`

> **Note:** `evaluate.py` imports `deepeval.login(settings.confident_api_key)` at module level. Importing from it will trigger this login call. This is acceptable.

> **Bug in existing code:** `evaluate.py:94` calls `BasicRAG().generate(collection_name=collection_name, ...)` but `BasicRAG.generate()` does not accept `collection_name`. The router passes `collection_name` to the `BasicRAG()` constructor instead and calls `generate()` without it.

**Step 1: Implement `src/api/routers/evaluation.py`**

```python
import asyncio
import json
from pathlib import Path
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.api.job_store import JobStore, JobStatus
from src.evaluation.evaluate import (
    create_metrics,
    evaluate_llm_test_case_on_metrics,
)
from src.retrieval.basic_rag import BasicRAG
from src.evaluation.evaluate import create_llm_test_case
from deepeval.test_case import LLMTestCase
from src.settings import settings

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
job_store = JobStore()


class EvaluationRequest(BaseModel):
    goldens_dir: str
    retrieval_window_size: int = 5
    collection_name: str = None
    threshold: float = 0.5
    force_rerun: bool = False


def _run_evaluation(job_id: str, request: EvaluationRequest) -> None:
    job_store.update(job_id, status=JobStatus.RUNNING)
    try:
        collection_name = request.collection_name or settings.qdrant_collection_name
        metric_wrappers = create_metrics(threshold=request.threshold)
        metrics_summary: dict[str, list] = {}
        evaluated = 0
        skipped = 0

        for file_path in Path(request.goldens_dir).glob("**/*.json"):
            with open(file_path, encoding="utf-8") as f:
                sample = json.load(f)

            if (
                not request.force_rerun
                and sample.get("actual_output") is not None
                and sample.get("metrics") is not None
            ):
                skipped += 1
                continue

            try:
                # create_llm_test_case internally calls BasicRAG().generate() which
                # is async — wrap in asyncio.run()
                rag = BasicRAG(qdrant_collection_name=collection_name)
                retrieval_infos = asyncio.run(
                    rag.retrieve(query=sample["input"], top_k=request.retrieval_window_size)
                )
                actual_output = asyncio.run(
                    rag.generate(query=sample["input"], top_k=request.retrieval_window_size)
                )
                test_case = LLMTestCase(
                    input=sample["input"],
                    expected_output=sample["expectedOutput"],
                    context=sample["context"],
                    actual_output=actual_output,
                    retrieval_context=[r.content for r in retrieval_infos],
                )
                metrics_result = evaluate_llm_test_case_on_metrics(test_case, metric_wrappers)
                sample["actual_output"] = actual_output
                sample["metrics"] = metrics_result
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(sample, f, indent=4)
                evaluated += 1

                for metric_name, data in metrics_result.items():
                    metrics_summary.setdefault(metric_name, []).append(data["score"])

            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
                continue

        avg_scores = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in metrics_summary.items()
            if scores
        }

        job_store.update(
            job_id,
            status=JobStatus.DONE,
            result={
                "evaluated": evaluated,
                "skipped": skipped,
                "avg_scores": avg_scores,
            },
        )
    except Exception as e:
        logger.exception(f"Evaluation job {job_id} failed: {e}")
        job_store.update(job_id, status=JobStatus.FAILED, error=str(e))


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_evaluation_job(
    body: EvaluationRequest, background_tasks: BackgroundTasks
):
    if not Path(body.goldens_dir).exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"goldens_dir does not exist: {body.goldens_dir}",
        )
    job_id = job_store.create()
    background_tasks.add_task(_run_evaluation, job_id, body)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_evaluation_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
        )
    return job
```

**Step 2: Register router in `src/api/app.py`**

Add import:
```python
from src.api.routers.evaluation import router as evaluation_router
```

Add in `create_app()`:
```python
app.include_router(evaluation_router, prefix=settings.api_prefix)
```

**Step 3: Smoke test**

```bash
curl -X POST http://localhost:8000/api/v1/evaluation/jobs \
  -H "Content-Type: application/json" \
  -d '{"goldens_dir": "data/goldens", "retrieval_window_size": 5}'
# Expected: {"job_id": "<uuid>"}
```

**Step 4: Commit**

```bash
git add src/api/routers/evaluation.py src/api/app.py
git commit -m "feat: add /evaluation router with background job support"
```

---

## Task 7: Shared Streamlit API client

**Files:**
- Create: `src/ui/__init__.py` (empty)
- Create: `src/ui/api_client.py`

**Step 1: Create `src/ui/__init__.py`**

Empty file — makes `src/ui` a proper package.

**Step 2: Create `src/ui/api_client.py`**

```python
import os
import httpx

BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"

# Synchronous client for most Streamlit calls
client = httpx.Client(base_url=BASE_URL, timeout=120.0)

# Streaming client (for Agent SSE)
stream_client = httpx.Client(base_url=BASE_URL, timeout=None)


def api_url(path: str) -> str:
    return f"{API_PREFIX}{path}"
```

**Step 3: Commit**

```bash
git add src/ui/__init__.py src/ui/api_client.py
git commit -m "feat: add shared httpx API client for Streamlit"
```

---

## Task 8: Streamlit Home page

**Files:**
- Create: `src/ui/Home.py`
- Create: `src/ui/pages/.gitkeep`

**Step 1: Create `src/ui/pages/` directory**

```bash
mkdir -p src/ui/pages
touch src/ui/pages/.gitkeep
```

**Step 2: Create `src/ui/Home.py`**

```python
import streamlit as st
from src.ui.api_client import client, BASE_URL

st.set_page_config(page_title="newbieAR Pipeline Explorer", layout="wide")

st.title("newbieAR — Pipeline Explorer")
st.caption("End-to-end Agentic RAG developer tool")

# Sidebar: connection settings
with st.sidebar:
    st.header("Backend")
    st.text_input(
        "FastAPI URL",
        value=BASE_URL,
        key="fastapi_url",
        help="Set FASTAPI_BASE_URL env var to change default",
        disabled=True,
    )
    try:
        r = client.get("/health")
        if r.status_code == 200:
            st.success("API: connected")
        else:
            st.warning(f"API: status {r.status_code}")
    except Exception as e:
        st.error(f"API: unreachable — {e}")

st.markdown(
    """
## Pipeline Stages

Use the sidebar to navigate between stages:

| Page | What it does |
|------|-------------|
| **1 — Ingestion** | Upload PDFs into Qdrant (vector) or Neo4j (graph) |
| **2 — Retrieval** | Run BasicRAG or GraphRAG queries |
| **3 — Agent** | Chat with the pydantic-ai agentic RAG |
| **4 — Synthesis** | Generate golden test cases with deepeval Synthesizer |
| **5 — Evaluation** | Score goldens with deepeval metrics (Bedrock critic) |
"""
)
```

**Step 3: Verify Streamlit runs**

```bash
uv run streamlit run src/ui/Home.py
```

Expected: browser opens, health check shows "connected" if FastAPI is running.

**Step 4: Commit**

```bash
git add src/ui/Home.py src/ui/pages/.gitkeep
git commit -m "feat: add Streamlit home page"
```

---

## Task 9: Ingestion page

**Files:**
- Create: `src/ui/pages/1_Ingestion.py`

**Step 1: Create `src/ui/pages/1_Ingestion.py`**

```python
import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Ingestion", layout="wide")
st.title("Ingestion")
st.caption("Upload a PDF into the vector DB (Qdrant) or graph DB (Neo4j)")

mode = st.radio("Target database", ["Vector DB", "Graph DB"], horizontal=True)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if mode == "Vector DB":
    col1, col2 = st.columns(2)
    with col1:
        collection_name = st.text_input("Collection name", value="research_papers")
    with col2:
        chunk_strategy = st.selectbox("Chunk strategy", ["hybrid", "hierarchical"])
else:
    chunk_strategy = st.selectbox("Chunk strategy", ["hierarchical", "hybrid"])

if st.button("Ingest", disabled=uploaded is None):
    with st.spinner("Ingesting..."):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        if mode == "Vector DB":
            data = {"collection_name": collection_name, "chunk_strategy": chunk_strategy}
            r = client.post(api_url("/ingest/vector"), data=data, files=files)
        else:
            data = {"chunk_strategy": chunk_strategy}
            r = client.post(api_url("/ingest/graph"), data=data, files=files)

    if r.status_code == 200:
        body = r.json()
        st.success("Ingestion complete!")
        st.json(body)
    else:
        st.error(f"Error {r.status_code}: {r.text}")
```

**Step 2: Verify page loads**

```bash
uv run streamlit run src/ui/Home.py
```

Navigate to "Ingestion" in the sidebar. Upload a PDF and click Ingest.

**Step 3: Commit**

```bash
git add src/ui/pages/1_Ingestion.py
git commit -m "feat: add Ingestion Streamlit page"
```

---

## Task 10: Retrieval page

**Files:**
- Create: `src/ui/pages/2_Retrieval.py`

**Step 1: Create `src/ui/pages/2_Retrieval.py`**

```python
import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Retrieval", layout="wide")
st.title("Retrieval")
st.caption("Query documents via BasicRAG (Qdrant) or GraphRAG (Neo4j)")

mode = st.radio("Retrieval mode", ["Vector (BasicRAG)", "Graph (GraphRAG)"], horizontal=True)

query = st.text_input("Query")

col1, col2, col3, col4 = st.columns(4)
with col1:
    top_k = st.number_input("top_k", min_value=1, max_value=50, value=5)
if mode == "Vector (BasicRAG)":
    with col2:
        collection_name = st.text_input("Collection name", value="research_papers")
    with col3:
        score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.0, step=0.05)
    with col4:
        rerank = st.toggle("Rerank (cross-encoder)")

if st.button("Search", disabled=not query.strip()):
    with st.spinner("Retrieving..."):
        if mode == "Vector (BasicRAG)":
            payload = {
                "query": query,
                "collection_name": collection_name,
                "top_k": int(top_k),
                "score_threshold": score_threshold,
                "rerank": rerank,
            }
            r = client.post(api_url("/retrieve/vector"), json=payload)
        else:
            r = client.post(
                api_url("/retrieve/graph"), json={"query": query, "top_k": int(top_k)}
            )

    if r.status_code == 200:
        body = r.json()
        st.success("Done!")
        if mode == "Vector (BasicRAG)":
            for i, result in enumerate(body.get("results", [])):
                with st.expander(f"[{i+1}] {result['source']} — score: {result['score']:.4f}"):
                    st.write(result["content"])
        else:
            for i, ctx in enumerate(body.get("contexts", [])):
                with st.expander(f"Context {i+1}"):
                    st.write(ctx)
            if body.get("citations"):
                st.caption("Citations: " + ", ".join(body["citations"]))
    else:
        st.error(f"Error {r.status_code}: {r.text}")
```

**Step 2: Verify and commit**

```bash
git add src/ui/pages/2_Retrieval.py
git commit -m "feat: add Retrieval Streamlit page"
```

---

## Task 11: Agent page

**Files:**
- Create: `src/ui/pages/3_Agent.py`

**Context:**
- Requires a session: `POST /api/v1/sessions` with `{collection_name, top_k}` → `{session_id}`
- Chat: `POST /api/v1/chat/stream` with `{session_id, message}` → SSE stream of `delta` events then `done`
- SSE with `httpx`: use `client.stream("POST", url, json=body)` and iterate `response.iter_lines()`

**Step 1: Create `src/ui/pages/3_Agent.py`**

```python
import json
import streamlit as st
from src.ui.api_client import client, stream_client, api_url

st.set_page_config(page_title="Agent", layout="wide")
st.title("Agentic RAG")
st.caption("Chat with the pydantic-ai agent (BasicRAG + GraphRAG tools)")

# Session management
with st.sidebar:
    st.subheader("Session")
    collection = st.text_input("Collection name", value="research_papers")
    top_k = st.number_input("top_k", min_value=1, max_value=50, value=5)

    if st.button("New session"):
        r = client.post(
            api_url("/sessions"),
            json={"collection_name": collection, "top_k": int(top_k)},
        )
        if r.status_code == 201:
            st.session_state["session_id"] = r.json()["session_id"]
            st.session_state["messages"] = []
            st.success(f"Session: {st.session_state['session_id'][:8]}...")
        else:
            st.error(f"Error: {r.text}")

    if "session_id" in st.session_state:
        st.caption(f"Active: `{st.session_state['session_id'][:8]}...`")
        if st.button("Delete session"):
            client.delete(api_url(f"/sessions/{st.session_state['session_id']}"))
            del st.session_state["session_id"]
            st.session_state["messages"] = []
            st.rerun()

# Chat area
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("contexts"):
            with st.expander("Contexts"):
                for ctx in msg["contexts"]:
                    st.write(ctx)
        if msg.get("citations"):
            st.caption("Citations: " + ", ".join(msg["citations"]))

if prompt := st.chat_input("Ask a question...", disabled="session_id" not in st.session_state):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        contexts = []
        citations = []

        try:
            with stream_client.stream(
                "POST",
                api_url("/chat/stream"),
                json={"session_id": st.session_state["session_id"], "message": prompt},
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        if event == "delta":
                            full_text += data.get("text", "")
                            placeholder.write(full_text)
                        elif event == "done":
                            contexts = data.get("contexts", [])
                            citations = data.get("citations", [])
                        elif event == "error":
                            st.error(data.get("detail", "Unknown error"))
        except Exception as e:
            st.error(f"Stream error: {e}")

        if contexts:
            with st.expander("Contexts"):
                for ctx in contexts:
                    st.write(ctx)
        if citations:
            st.caption("Citations: " + ", ".join(citations))

    st.session_state["messages"].append(
        {"role": "assistant", "content": full_text, "contexts": contexts, "citations": citations}
    )
```

**Step 2: Verify and commit**

```bash
git add src/ui/pages/3_Agent.py
git commit -m "feat: add Agent Streamlit page with SSE streaming"
```

---

## Task 12: Synthesis page

**Files:**
- Create: `src/ui/pages/4_Synthesis.py`

**Step 1: Create `src/ui/pages/4_Synthesis.py`**

```python
import time
import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Synthesis", layout="wide")
st.title("Synthesis")
st.caption("Generate golden test cases from documents using deepeval Synthesizer (AWS Bedrock)")

col1, col2 = st.columns(2)
with col1:
    file_dir = st.text_input("Source file directory", value="data/papers/files")
    output_dir = st.text_input("Output directory", value="data/goldens")
with col2:
    topic = st.selectbox("Topic", ["paper", "article"])
    num_contexts = st.number_input("Contexts per file", min_value=1, max_value=50, value=5)
    context_size = st.number_input("Chunks per context", min_value=1, max_value=10, value=5)

if st.button("Start Synthesis"):
    r = client.post(
        api_url("/synthesis/jobs"),
        json={
            "file_dir": file_dir,
            "output_dir": output_dir,
            "topic": topic,
            "num_contexts": int(num_contexts),
            "context_size": int(context_size),
        },
    )
    if r.status_code == 202:
        st.session_state["synthesis_job_id"] = r.json()["job_id"]
        st.session_state["synthesis_done"] = False
    else:
        st.error(f"Error {r.status_code}: {r.text}")

if job_id := st.session_state.get("synthesis_job_id"):
    if not st.session_state.get("synthesis_done"):
        with st.spinner(f"Running synthesis job `{job_id[:8]}...`"):
            while True:
                r = client.get(api_url(f"/synthesis/jobs/{job_id}"))
                job = r.json()
                if job["status"] in ("done", "failed"):
                    st.session_state["synthesis_done"] = True
                    break
                time.sleep(3)

    r = client.get(api_url(f"/synthesis/jobs/{job_id}"))
    job = r.json()
    if job["status"] == "done":
        st.success(f"Done! Generated {job['result']['goldens_count']} goldens.")
        st.json(job["result"])
    elif job["status"] == "failed":
        st.error(f"Job failed: {job.get('error')}")
    else:
        st.info(f"Status: {job['status']}")
```

**Step 2: Verify and commit**

```bash
git add src/ui/pages/4_Synthesis.py
git commit -m "feat: add Synthesis Streamlit page with job polling"
```

---

## Task 13: Evaluation page

**Files:**
- Create: `src/ui/pages/5_Evaluation.py`

**Step 1: Create `src/ui/pages/5_Evaluation.py`**

```python
import time
import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation")
st.caption("Score golden test cases with deepeval metrics (Bedrock critique model)")

col1, col2 = st.columns(2)
with col1:
    goldens_dir = st.text_input("Goldens directory", value="data/goldens")
    collection_name = st.text_input("Qdrant collection name", value="research_papers")
with col2:
    retrieval_window_size = st.number_input("Retrieval window size (top_k)", min_value=1, max_value=20, value=5)
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, step=0.05)
    force_rerun = st.toggle("Force re-evaluation (skip cache)")

if st.button("Start Evaluation"):
    r = client.post(
        api_url("/evaluation/jobs"),
        json={
            "goldens_dir": goldens_dir,
            "retrieval_window_size": int(retrieval_window_size),
            "collection_name": collection_name,
            "threshold": threshold,
            "force_rerun": force_rerun,
        },
    )
    if r.status_code == 202:
        st.session_state["eval_job_id"] = r.json()["job_id"]
        st.session_state["eval_done"] = False
    else:
        st.error(f"Error {r.status_code}: {r.text}")

if job_id := st.session_state.get("eval_job_id"):
    if not st.session_state.get("eval_done"):
        with st.spinner(f"Running evaluation job `{job_id[:8]}...` — this may take several minutes"):
            while True:
                r = client.get(api_url(f"/evaluation/jobs/{job_id}"))
                job = r.json()
                if job["status"] in ("done", "failed"):
                    st.session_state["eval_done"] = True
                    break
                time.sleep(3)

    r = client.get(api_url(f"/evaluation/jobs/{job_id}"))
    job = r.json()
    if job["status"] == "done":
        result = job["result"]
        st.success(
            f"Done! Evaluated {result['evaluated']} cases, "
            f"skipped {result['skipped']} cached."
        )

        avg = result.get("avg_scores", {})
        if avg:
            st.subheader("Average Scores")
            cols = st.columns(len(avg))
            for col, (name, score) in zip(cols, avg.items()):
                label = name.replace("Metric", "").strip()
                delta_color = "normal" if score >= threshold else "inverse"
                col.metric(label=label, value=f"{score:.3f}", delta=f"{'PASS' if score >= threshold else 'FAIL'}", delta_color=delta_color)

        st.json(result)
    elif job["status"] == "failed":
        st.error(f"Job failed: {job.get('error')}")
    else:
        st.info(f"Status: {job['status']}")
```

**Step 2: Verify and commit**

```bash
git add src/ui/pages/5_Evaluation.py
git commit -m "feat: add Evaluation Streamlit page with job polling and metrics display"
```

---

## Task 14: Final wiring and run all tests

**Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests pass; new tests (job_store, ingest, retrieve) pass.

**Step 2: Start the full stack and do an end-to-end smoke test**

Terminal 1 — FastAPI:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — Streamlit:
```bash
uv run streamlit run src/ui/Home.py
```

Check each page loads and the health indicator shows "connected".

**Step 3: Update CLAUDE.md with new commands**

In `docs/CLAUDE.md`, add under "Common Commands":
```markdown
### Run Streamlit UI
```bash
# Terminal 1: FastAPI backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit
uv run streamlit run src/ui/Home.py
```
```

**Step 4: Final commit**

```bash
git add docs/CLAUDE.md
git commit -m "docs: add Streamlit launch commands to CLAUDE.md"
```
