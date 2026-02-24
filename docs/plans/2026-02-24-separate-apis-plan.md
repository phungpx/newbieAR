# Separate APIs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three domain REST API routers (ingestion, retrieval, agents) to a FastAPI app, exposing existing VectorDBIngestion, BasicRAG, GraphRetrieval, and pydantic-ai agents over HTTP.

**Architecture:** Single FastAPI app at `src/api/app.py` with three domain packages under `src/api/`; each domain has `router.py`, `schemas.py`, `service.py`, and a store module. Ingestion runs as async background tasks with job polling. Agents use SSE streaming with Redis-backed session history. All stores default to in-memory; controlled by `settings.job_storage` / `settings.session_storage`.

**Tech Stack:** FastAPI, sse-starlette, python-multipart, pydantic-ai, httpx + pytest-asyncio (tests), fakeredis (optional).

---

### Task 1: API skeleton

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/app.py`
- Create: `src/api/deps.py`
- Create: `src/api/ingestion/__init__.py`
- Create: `src/api/retrieval/__init__.py`
- Create: `src/api/agents/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/api/__init__.py`

**Step 1: Create all empty `__init__.py` files**

```bash
mkdir -p src/api/ingestion src/api/retrieval src/api/agents tests/api
touch src/api/__init__.py src/api/ingestion/__init__.py src/api/retrieval/__init__.py src/api/agents/__init__.py
touch tests/__init__.py tests/api/__init__.py
```

**Step 2: Create `src/api/app.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="newbieAR API", version="1.0.0", lifespan=lifespan)
    return app


app = create_app()
```

**Step 3: Create `src/api/deps.py`**

```python
from src.api.ingestion.job_store import InMemoryJobStore
from src.api.agents.session_store import InMemorySessionStore

_job_store: InMemoryJobStore | None = None
_session_store: InMemorySessionStore | None = None


def get_job_store() -> InMemoryJobStore:
    global _job_store
    if _job_store is None:
        _job_store = InMemoryJobStore()
    return _job_store


def get_session_store() -> InMemorySessionStore:
    global _session_store
    if _session_store is None:
        _session_store = InMemorySessionStore()
    return _session_store
```

Note: `deps.py` imports from `job_store` and `session_store` which don't exist yet — that's fine, they'll be created in subsequent tasks. Don't run the app until Task 10.

**Step 4: Verify the skeleton is importable**

```bash
uv run python -c "from src.api.app import app; print('OK')"
```

Expected: `ModuleNotFoundError` for `job_store` — this is expected, proves skeleton is wired.

**Step 5: Commit**

```bash
git add src/api/ tests/
git commit -m "feat: scaffold api package skeleton"
```

---

### Task 2: Ingestion job store + schemas

**Files:**
- Create: `src/api/ingestion/job_store.py`
- Create: `src/api/ingestion/schemas.py`
- Create: `tests/api/test_ingestion_job_store.py`

**Step 1: Write the failing test**

```python
# tests/api/test_ingestion_job_store.py
import pytest
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus


def test_create_job_returns_pending():
    store = InMemoryJobStore()
    job = store.create_job()
    assert job.status == JobStatus.PENDING
    assert job.job_id


def test_get_job_returns_none_for_unknown():
    store = InMemoryJobStore()
    assert store.get_job("nonexistent") is None


def test_update_job_status():
    store = InMemoryJobStore()
    job = store.create_job()
    updated = store.update_job(job.job_id, JobStatus.RUNNING)
    assert updated.status == JobStatus.RUNNING


def test_update_job_done_with_result():
    store = InMemoryJobStore()
    job = store.create_job()
    result = {"chunks_count": 5, "collection_name": "test"}
    updated = store.update_job(job.job_id, JobStatus.DONE, result=result)
    assert updated.status == JobStatus.DONE
    assert updated.result == result


def test_update_job_failed_with_error():
    store = InMemoryJobStore()
    job = store.create_job()
    updated = store.update_job(job.job_id, JobStatus.FAILED, error="something went wrong")
    assert updated.status == JobStatus.FAILED
    assert updated.error == "something went wrong"
```

**Step 2: Run test to confirm failure**

```bash
uv run pytest tests/api/test_ingestion_job_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.ingestion.job_store'`

**Step 3: Write `src/api/ingestion/job_store.py`**

```python
import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        self._jobs[job.job_id] = job
        return job

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> Job:
        job = self._jobs[job_id]
        job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
```

**Step 4: Write `src/api/ingestion/schemas.py`**

```python
from typing import Optional
from pydantic import BaseModel
from src.api.ingestion.job_store import JobStatus


class IngestionJobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[dict] = None
    error: Optional[str] = None
```

**Step 5: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_ingestion_job_store.py -v
```

Expected: 5 PASSED

**Step 6: Commit**

```bash
git add src/api/ingestion/ tests/api/test_ingestion_job_store.py
git commit -m "feat: add ingestion job store and schemas"
```

---

### Task 3: Ingestion service

**Files:**
- Create: `src/api/ingestion/service.py`
- Create: `tests/api/test_ingestion_service.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_ingestion_service.py
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.api.ingestion.service import IngestionService


@pytest.fixture
def job_store():
    return InMemoryJobStore()


@pytest.fixture
def service(job_store):
    return IngestionService(job_store)


@pytest.mark.asyncio
async def test_ingest_vectordb_returns_job_id(service, job_store):
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_file.return_value = {
        "file_save_path": "/tmp/doc.md",
        "chunk_save_path": "/tmp/chunks.json",
        "qdrant_collection_name": "test",
    }
    with patch("src.api.ingestion.service.VectorDBIngestion", return_value=mock_ingestion):
        job_id = await service.ingest_vectordb(
            file_bytes=b"PDF content",
            filename="test.pdf",
            collection_name="test",
            chunk_strategy="hybrid",
        )
    assert job_id is not None
    # Give background task time to run
    await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.DONE
    assert job.result["qdrant_collection_name"] == "test"


@pytest.mark.asyncio
async def test_ingest_vectordb_marks_failed_on_error(service, job_store):
    with patch("src.api.ingestion.service.VectorDBIngestion", side_effect=Exception("connection error")):
        job_id = await service.ingest_vectordb(
            file_bytes=b"data",
            filename="doc.pdf",
            collection_name="col",
            chunk_strategy="hybrid",
        )
    await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert "connection error" in job.error


@pytest.mark.asyncio
async def test_ingest_graphdb_returns_job_id(service, job_store):
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_file = asyncio.coroutine(lambda path: None)

    async def mock_ingest_file(path):
        pass

    mock_ingestion.ingest_file = mock_ingest_file
    mock_ingestion.close = asyncio.coroutine(lambda: None)

    async def mock_close():
        pass

    mock_ingestion.close = mock_close

    with patch("src.api.ingestion.service.GraphitiIngestion", return_value=mock_ingestion):
        job_id = await service.ingest_graphdb(
            file_bytes=b"data",
            filename="doc.pdf",
            chunk_strategy="hierarchical",
        )
    await asyncio.sleep(0.1)
    job = job_store.get_job(job_id)
    assert job.status == JobStatus.DONE
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_ingestion_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.ingestion.service'`

**Step 3: Write `src/api/ingestion/service.py`**

```python
import asyncio
import os
import tempfile
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.ingestion.ingest_graphdb import GraphitiIngestion
from src.models import ChunkStrategy


class IngestionService:
    def __init__(self, job_store: InMemoryJobStore):
        self.job_store = job_store

    async def ingest_vectordb(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        chunk_strategy: str,
    ) -> str:
        job = self.job_store.create_job()
        asyncio.create_task(
            self._run_vectordb(job.job_id, file_bytes, filename, collection_name, chunk_strategy)
        )
        return job.job_id

    async def _run_vectordb(
        self,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        chunk_strategy: str,
    ):
        self.job_store.update_job(job_id, JobStatus.RUNNING)
        tmp_path = None
        try:
            suffix = os.path.splitext(filename)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            ingestion = VectorDBIngestion(
                documents_dir="data/papers/docs",
                chunks_dir="data/papers/chunks",
                chunk_strategy=chunk_strategy,
                qdrant_collection_name=collection_name,
            )
            result = ingestion.ingest_file(tmp_path)
            self.job_store.update_job(job_id, JobStatus.DONE, result=result)
        except Exception as e:
            self.job_store.update_job(job_id, JobStatus.FAILED, error=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def ingest_graphdb(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_strategy: str,
    ) -> str:
        job = self.job_store.create_job()
        asyncio.create_task(
            self._run_graphdb(job.job_id, file_bytes, filename, chunk_strategy)
        )
        return job.job_id

    async def _run_graphdb(
        self,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        chunk_strategy: str,
    ):
        self.job_store.update_job(job_id, JobStatus.RUNNING)
        tmp_path = None
        ingestion = None
        try:
            suffix = os.path.splitext(filename)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            ingestion = GraphitiIngestion(chunk_strategy=chunk_strategy)
            await ingestion.ingest_file(tmp_path)
            self.job_store.update_job(job_id, JobStatus.DONE, result={"filename": filename})
        except Exception as e:
            self.job_store.update_job(job_id, JobStatus.FAILED, error=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if ingestion:
                await ingestion.close()
```

**Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_ingestion_service.py -v
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/api/ingestion/service.py tests/api/test_ingestion_service.py
git commit -m "feat: add ingestion service with async job execution"
```

---

### Task 4: Ingestion router

**Files:**
- Create: `src/api/ingestion/router.py`
- Create: `tests/api/test_ingestion_router.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_ingestion_router.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.ingestion.router import router
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.api import deps


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def job_store():
    return InMemoryJobStore()


@pytest.fixture
def test_app_with_store(test_app, job_store):
    test_app.dependency_overrides[deps.get_job_store] = lambda: job_store
    yield test_app
    test_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ingest_vectordb_returns_202(test_app_with_store):
    with patch("src.api.ingestion.service.VectorDBIngestion") as mock_cls:
        mock_cls.return_value.ingest_file.return_value = {
            "qdrant_collection_name": "test",
            "file_save_path": "/tmp/a",
            "chunk_save_path": "/tmp/b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=test_app_with_store), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingestion/vectordb",
                data={"collection_name": "test", "chunk_strategy": "hybrid"},
                files={"file": ("doc.pdf", b"PDF content", "application/pdf")},
            )
    assert response.status_code == 202
    assert "job_id" in response.json()


@pytest.mark.asyncio
async def test_get_job_status_not_found(test_app_with_store):
    async with AsyncClient(
        transport=ASGITransport(app=test_app_with_store), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/ingestion/jobs/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_status_found(test_app_with_store, job_store):
    job = job_store.create_job()
    job_store.update_job(job.job_id, JobStatus.DONE, result={"qdrant_collection_name": "test"})
    async with AsyncClient(
        transport=ASGITransport(app=test_app_with_store), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/ingestion/jobs/{job.job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert data["result"]["qdrant_collection_name"] == "test"


@pytest.mark.asyncio
async def test_ingest_graphdb_returns_202(test_app_with_store):
    async def mock_ingest_file(path):
        pass

    async def mock_close():
        pass

    mock_obj = AsyncMock()
    mock_obj.ingest_file = mock_ingest_file
    mock_obj.close = mock_close

    with patch("src.api.ingestion.service.GraphitiIngestion", return_value=mock_obj):
        async with AsyncClient(
            transport=ASGITransport(app=test_app_with_store), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/ingestion/graphdb",
                data={"chunk_strategy": "hierarchical"},
                files={"file": ("doc.pdf", b"PDF content", "application/pdf")},
            )
    assert response.status_code == 202
    assert "job_id" in response.json()
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_ingestion_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.ingestion.router'`

**Step 3: Write `src/api/ingestion/router.py`**

```python
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from src.api import deps
from src.api.ingestion.job_store import InMemoryJobStore
from src.api.ingestion.schemas import IngestionJobResponse, JobStatusResponse
from src.api.ingestion.service import IngestionService
from src.models import ChunkStrategy

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/vectordb", status_code=202, response_model=IngestionJobResponse)
async def ingest_vectordb(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_strategy: str = Form(ChunkStrategy.HYBRID.value),
    job_store: InMemoryJobStore = Depends(deps.get_job_store),
):
    service = IngestionService(job_store)
    file_bytes = await file.read()
    job_id = await service.ingest_vectordb(
        file_bytes=file_bytes,
        filename=file.filename or "upload.pdf",
        collection_name=collection_name,
        chunk_strategy=chunk_strategy,
    )
    return IngestionJobResponse(job_id=job_id)


@router.post("/graphdb", status_code=202, response_model=IngestionJobResponse)
async def ingest_graphdb(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(ChunkStrategy.HIERARCHICAL.value),
    job_store: InMemoryJobStore = Depends(deps.get_job_store),
):
    service = IngestionService(job_store)
    file_bytes = await file.read()
    job_id = await service.ingest_graphdb(
        file_bytes=file_bytes,
        filename=file.filename or "upload.pdf",
        chunk_strategy=chunk_strategy,
    )
    return IngestionJobResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    job_store: InMemoryJobStore = Depends(deps.get_job_store),
):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job.model_dump())
```

**Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_ingestion_router.py -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/api/ingestion/router.py tests/api/test_ingestion_router.py
git commit -m "feat: add ingestion router with vectordb, graphdb, and job polling endpoints"
```

---

### Task 5: Retrieval schemas + service

**Files:**
- Create: `src/api/retrieval/schemas.py`
- Create: `src/api/retrieval/service.py`
- Create: `tests/api/test_retrieval_service.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_retrieval_service.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.api.retrieval.service import RetrievalService
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


@pytest.fixture
def service():
    return RetrievalService()


def test_retrieve_basic_returns_retrieval_infos(service):
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        RetrievalInfo(content="doc text", source="file.pdf - Chunk #1", score=0.9)
    ]
    with patch("src.api.retrieval.service.BasicRAG", return_value=mock_rag):
        results = service.retrieve_basic(query="test", collection_name="col", top_k=3)
    assert len(results) == 1
    assert results[0].content == "doc text"
    mock_rag.retrieve.assert_called_once_with("test", top_k=3)


@pytest.mark.asyncio
async def test_retrieve_graph_returns_node_edge_episode(service):
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve.return_value = (
        [GraphitiNodeInfo(uuid="n1", summary="Node 1")],
        [GraphitiEdgeInfo(uuid="e1", fact="Fact 1")],
        [GraphitiEpisodeInfo(uuid="ep1", content="Episode 1")],
    )
    mock_retrieval.close = AsyncMock()
    with patch("src.api.retrieval.service.GraphRetrieval", return_value=mock_retrieval):
        nodes, edges, episodes = await service.retrieve_graph(query="test", top_k=5)
    assert len(nodes) == 1
    assert nodes[0].uuid == "n1"
    assert len(edges) == 1
    assert len(episodes) == 1
    mock_retrieval.close.assert_awaited_once()
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_retrieval_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.retrieval.service'`

**Step 3: Write `src/api/retrieval/schemas.py`**

```python
from pydantic import BaseModel
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


class BasicRetrievalRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5


class BasicRetrievalResponse(BaseModel):
    results: list[RetrievalInfo]


class GraphRetrievalRequest(BaseModel):
    query: str
    top_k: int = 5


class GraphRetrievalResponse(BaseModel):
    nodes: list[GraphitiNodeInfo]
    edges: list[GraphitiEdgeInfo]
    episodes: list[GraphitiEpisodeInfo]
```

**Step 4: Write `src/api/retrieval/service.py`**

```python
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


class RetrievalService:
    def retrieve_basic(
        self, query: str, collection_name: str, top_k: int
    ) -> list[RetrievalInfo]:
        rag = BasicRAG(qdrant_collection_name=collection_name)
        return rag.retrieve(query, top_k=top_k)

    async def retrieve_graph(
        self, query: str, top_k: int
    ) -> tuple[
        list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]
    ]:
        retrieval = GraphRetrieval()
        try:
            nodes, edges, episodes = await retrieval.retrieve(query, num_results=top_k)
            return nodes, edges, episodes
        finally:
            await retrieval.close()
```

**Step 5: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_retrieval_service.py -v
```

Expected: 2 PASSED

**Step 6: Commit**

```bash
git add src/api/retrieval/ tests/api/test_retrieval_service.py
git commit -m "feat: add retrieval schemas and service"
```

---

### Task 6: Retrieval router

**Files:**
- Create: `src/api/retrieval/router.py`
- Create: `tests/api/test_retrieval_router.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_retrieval_router.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.retrieval.router import router
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_retrieve_basic_returns_results(test_app):
    mock_service = MagicMock()
    mock_service.retrieve_basic.return_value = [
        RetrievalInfo(content="doc", source="file.pdf - Chunk #1", score=0.95)
    ]
    with patch("src.api.retrieval.router.RetrievalService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/basic",
                json={"query": "what is RAG?", "collection_name": "papers", "top_k": 3},
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "doc"


@pytest.mark.asyncio
async def test_retrieve_graph_returns_results(test_app):
    mock_service = AsyncMock()
    mock_service.retrieve_graph.return_value = (
        [GraphitiNodeInfo(uuid="n1", summary="Node 1")],
        [GraphitiEdgeInfo(uuid="e1", fact="Fact 1")],
        [GraphitiEpisodeInfo(uuid="ep1", content="Episode 1")],
    )
    with patch("src.api.retrieval.router.RetrievalService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/graph",
                json={"query": "who is the author?", "top_k": 5},
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["uuid"] == "n1"
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_retrieval_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.retrieval.router'`

**Step 3: Write `src/api/retrieval/router.py`**

```python
from fastapi import APIRouter
from src.api.retrieval.schemas import (
    BasicRetrievalRequest,
    BasicRetrievalResponse,
    GraphRetrievalRequest,
    GraphRetrievalResponse,
)
from src.api.retrieval.service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/basic", response_model=BasicRetrievalResponse)
async def retrieve_basic(request: BasicRetrievalRequest):
    service = RetrievalService()
    results = service.retrieve_basic(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )
    return BasicRetrievalResponse(results=results)


@router.post("/graph", response_model=GraphRetrievalResponse)
async def retrieve_graph(request: GraphRetrievalRequest):
    service = RetrievalService()
    nodes, edges, episodes = await service.retrieve_graph(
        query=request.query, top_k=request.top_k
    )
    return GraphRetrievalResponse(nodes=nodes, edges=edges, episodes=episodes)
```

**Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_retrieval_router.py -v
```

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/api/retrieval/router.py tests/api/test_retrieval_router.py
git commit -m "feat: add retrieval router for basic and graph endpoints"
```

---

### Task 7: Agent session store + schemas

**Files:**
- Create: `src/api/agents/session_store.py`
- Create: `src/api/agents/schemas.py`
- Create: `tests/api/test_agent_session_store.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_agent_session_store.py
import pytest
from src.api.agents.session_store import InMemorySessionStore


def test_get_or_create_new_session():
    store = InMemorySessionStore()
    session_id, messages = store.get_or_create(None)
    assert session_id is not None
    assert messages == []


def test_get_or_create_existing_session():
    store = InMemorySessionStore()
    session_id, _ = store.get_or_create(None)
    store.save(session_id, [{"role": "user", "content": "hi"}])
    same_id, messages = store.get_or_create(session_id)
    assert same_id == session_id
    assert len(messages) == 1


def test_get_or_create_unknown_session_id_creates_new():
    store = InMemorySessionStore()
    new_id, messages = store.get_or_create("does-not-exist")
    assert new_id != "does-not-exist"
    assert messages == []


def test_delete_existing_session():
    store = InMemorySessionStore()
    session_id, _ = store.get_or_create(None)
    result = store.delete(session_id)
    assert result is True
    _, messages = store.get_or_create(session_id)
    assert messages == []


def test_delete_nonexistent_session_returns_false():
    store = InMemorySessionStore()
    assert store.delete("ghost") is False
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_agent_session_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.agents.session_store'`

**Step 3: Write `src/api/agents/session_store.py`**

```python
import uuid
from typing import Any, Optional


class InMemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, list[Any]] = {}

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, list[Any]]:
        if session_id and session_id in self._sessions:
            return session_id, list(self._sessions[session_id])
        new_id = str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id, []

    def save(self, session_id: str, messages: list[Any]):
        self._sessions[session_id] = list(messages)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
```

**Step 4: Write `src/api/agents/schemas.py`**

```python
from typing import Optional
from pydantic import BaseModel


class AgentRequest(BaseModel):
    query: str
    collection_name: str = "research_papers"
    top_k: int = 5
    session_id: Optional[str] = None


class GraphAgentRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None
```

**Step 5: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_agent_session_store.py -v
```

Expected: 5 PASSED

**Step 6: Commit**

```bash
git add src/api/agents/ tests/api/test_agent_session_store.py
git commit -m "feat: add agent session store and schemas"
```

---

### Task 8: Agent service

**Files:**
- Create: `src/api/agents/service.py`
- Create: `tests/api/test_agent_service.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_agent_service.py
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.agents.session_store import InMemorySessionStore
from src.api.agents.service import AgentService


@pytest.fixture
def session_store():
    return InMemorySessionStore()


@pytest.fixture
def service(session_store):
    return AgentService(session_store)


@pytest.mark.asyncio
async def test_stream_basic_yields_tokens_and_done(service):
    mock_result = AsyncMock()
    mock_result.stream_text.return_value = _async_iter(["Hello", " world"])
    mock_result.all_messages.return_value = []

    async def mock_run_stream(*args, **kwargs):
        class Ctx:
            async def __aenter__(self):
                return mock_result
            async def __aexit__(self, *a):
                pass
        return Ctx()

    mock_agent = MagicMock()
    mock_agent.run_stream = mock_run_stream

    mock_rag = MagicMock()

    with patch("src.api.agents.service.basic_rag_agent", mock_agent), \
         patch("src.api.agents.service.BasicRAG", return_value=mock_rag), \
         patch("src.api.agents.service.get_openai_model", return_value=MagicMock()):
        chunks = []
        async for chunk in service.stream_basic("what is RAG?", "papers", 3, None):
            chunks.append(chunk)

    events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]
    assert len(token_events) == 2
    assert token_events[0]["content"] == "Hello"
    assert len(done_events) == 1
    assert "session_id" in done_events[0]


@pytest.mark.asyncio
async def test_stream_basic_yields_error_on_exception(service):
    async def bad_run_stream(*args, **kwargs):
        class Ctx:
            async def __aenter__(self):
                raise RuntimeError("model down")
            async def __aexit__(self, *a):
                pass
        return Ctx()

    mock_agent = MagicMock()
    mock_agent.run_stream = bad_run_stream

    with patch("src.api.agents.service.basic_rag_agent", mock_agent), \
         patch("src.api.agents.service.BasicRAG", return_value=MagicMock()), \
         patch("src.api.agents.service.get_openai_model", return_value=MagicMock()):
        chunks = []
        async for chunk in service.stream_basic("q", "col", 3, None):
            chunks.append(chunk)

    events = [json.loads(c.removeprefix("data: ").strip()) for c in chunks if c.strip()]
    assert any(e["type"] == "error" for e in events)


async def _async_iter(items):
    for item in items:
        yield item
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_agent_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.agents.service'`

**Step 3: Write `src/api/agents/service.py`**

```python
import json
from typing import AsyncIterator, Optional
from src.api.agents.session_store import InMemorySessionStore
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies, get_openai_model
from src.agents.agentic_graph_rag import graphiti_agent, GraphitiDependencies
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval


class AgentService:
    def __init__(self, session_store: InMemorySessionStore):
        self.session_store = session_store

    async def stream_basic(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        session_id: Optional[str],
    ) -> AsyncIterator[str]:
        session_id, messages = self.session_store.get_or_create(session_id)
        model = get_openai_model()
        basic_rag = BasicRAG(qdrant_collection_name=collection_name)
        deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=top_k)

        try:
            async with await basic_rag_agent.run_stream(
                query, model=model, message_history=messages, deps=deps
            ) as result:
                async for token in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            self.session_store.save(session_id, result.all_messages())
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'citations': []})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def stream_graph(
        self,
        query: str,
        top_k: int,
        session_id: Optional[str],
    ) -> AsyncIterator[str]:
        session_id, messages = self.session_store.get_or_create(session_id)
        model = get_openai_model()
        graph_retrieval = GraphRetrieval()
        deps = GraphitiDependencies(graph_retrieval=graph_retrieval, top_k=top_k)

        try:
            async with await graphiti_agent.run_stream(
                query, model=model, message_history=messages, deps=deps
            ) as result:
                async for token in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            self.session_store.save(session_id, result.all_messages())
            citations = deps.citations or []
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'citations': citations})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            await graph_retrieval.close()
```

**Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_agent_service.py -v
```

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/api/agents/service.py tests/api/test_agent_service.py
git commit -m "feat: add agent service with SSE streaming for basic and graph agents"
```

---

### Task 9: Agents router

**Files:**
- Create: `src/api/agents/router.py`
- Create: `tests/api/test_agents_router.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_agents_router.py
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from src.api.agents.router import router
from src.api.agents.session_store import InMemorySessionStore
from src.api import deps


@pytest.fixture
def session_store():
    return InMemorySessionStore()


@pytest.fixture
def test_app(session_store):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_session_store] = lambda: session_store
    return app


@pytest.mark.asyncio
async def test_agent_basic_streams_sse(test_app):
    async def fake_stream(*args, **kwargs):
        yield 'data: {"type": "token", "content": "Hello"}\n\n'
        yield 'data: {"type": "done", "session_id": "abc", "citations": []}\n\n'

    mock_service = MagicMock()
    mock_service.stream_basic.return_value = fake_stream()

    with patch("src.api.agents.router.AgentService", return_value=mock_service):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/agents/basic",
                json={"query": "what is RAG?", "collection_name": "papers"},
            )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_delete_session_success(test_app, session_store):
    session_id, _ = session_store.get_or_create(None)
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/api/v1/agents/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


@pytest.mark.asyncio
async def test_delete_session_not_found(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.delete("/api/v1/agents/sessions/ghost-id")
    assert response.status_code == 404
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_agents_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api.agents.router'`

**Step 3: Write `src/api/agents/router.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.api import deps
from src.api.agents.schemas import AgentRequest, GraphAgentRequest
from src.api.agents.service import AgentService
from src.api.agents.session_store import InMemorySessionStore

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/basic")
async def agent_basic(
    request: AgentRequest,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    service = AgentService(session_store)
    return StreamingResponse(
        service.stream_basic(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
    )


@router.post("/graph")
async def agent_graph(
    request: GraphAgentRequest,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    service = AgentService(session_store)
    return StreamingResponse(
        service.stream_graph(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}
```

**Step 4: Run tests to confirm passing**

```bash
uv run pytest tests/api/test_agents_router.py -v
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/api/agents/router.py tests/api/test_agents_router.py
git commit -m "feat: add agents router with SSE streaming and session management"
```

---

### Task 10: Wire app.py + full test suite

**Files:**
- Modify: `src/api/app.py`
- Create: `tests/api/test_app.py`

**Step 1: Write the integration test**

```python
# tests/api/test_app.py
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.app import app


@pytest.mark.asyncio
async def test_docs_accessible():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ingestion_job_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/ingestion/jobs/nonexistent-job")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/v1/agents/sessions/nonexistent")
    assert response.status_code == 404
```

**Step 2: Run to confirm failure**

```bash
uv run pytest tests/api/test_app.py -v
```

Expected: 404 or routing errors because routers aren't registered yet.

**Step 3: Update `src/api/app.py` to register all routers**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.ingestion.router import router as ingestion_router
from src.api.retrieval.router import router as retrieval_router
from src.api.agents.router import router as agents_router
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="newbieAR API", version="1.0.0", lifespan=lifespan)
    prefix = settings.api_prefix  # "/api/v1"
    app.include_router(ingestion_router, prefix=prefix)
    app.include_router(retrieval_router, prefix=prefix)
    app.include_router(agents_router, prefix=prefix)
    return app


app = create_app()
```

**Step 4: Run all tests**

```bash
uv run pytest tests/api/ -v
```

Expected: All tests PASSED

**Step 5: Verify app starts**

```bash
uv run uvicorn src.api.app:app --port 8001 --reload &
sleep 2
curl http://localhost:8001/docs
kill %1
```

Expected: HTML page for Swagger UI

**Step 6: Final commit**

```bash
git add src/api/app.py tests/api/test_app.py
git commit -m "feat: wire all routers into app — ingestion, retrieval, agents APIs complete"
```

---

## Run All Tests

```bash
uv run pytest tests/api/ -v --tb=short
```

Expected output:
```
tests/api/test_ingestion_job_store.py   5 passed
tests/api/test_ingestion_service.py     3 passed
tests/api/test_ingestion_router.py      4 passed
tests/api/test_retrieval_service.py     2 passed
tests/api/test_retrieval_router.py      2 passed
tests/api/test_agent_session_store.py   5 passed
tests/api/test_agent_service.py         2 passed
tests/api/test_agents_router.py         3 passed
tests/api/test_app.py                   3 passed
29 passed
```
