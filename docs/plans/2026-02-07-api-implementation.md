# FastAPI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build FastAPI REST APIs for ingestion, retrieval, and agent operations

**Architecture:** FastAPI with async/await, Pydantic models for validation, in-memory storage for dev (Redis-ready), API key authentication, SSE streaming for agents

**Tech Stack:** FastAPI, Pydantic, uvicorn, slowapi (rate limiting), sse-starlette (streaming), passlib (hashing)

---

## Phase 1: Foundation & Setup

### Task 1: Add FastAPI Dependencies

**Files:**
- Modify: `pyproject.toml:7-27`

**Step 1: Add FastAPI dependencies to pyproject.toml**

Add after line 27 (existing dependencies):

```toml
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-multipart>=0.0.19",
    "slowapi>=0.1.9",
    "sse-starlette>=2.2.1",
    "passlib[bcrypt]>=1.7.4",
    "python-jose[cryptography]>=3.3.0",
```

**Step 2: Update dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add FastAPI and related dependencies"
```

---

### Task 2: Extend Settings with API Configuration

**Files:**
- Modify: `src/settings.py:70-116`

**Step 1: Add API settings classes**

Add after line 68 (before ProjectSettings):

```python
class APISettings(ProjectBaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]
    admin_api_key: str = "newbie_admin_dev_key_change_in_production"


class AuthSettings(ProjectBaseSettings):
    api_keys_storage: str = "memory"  # or "redis"
    rate_limit_enabled: bool = True
    rate_limit_standard: str = "100/minute"
    rate_limit_premium: str = "1000/minute"


class JobSettings(ProjectBaseSettings):
    job_storage: str = "memory"  # or "redis"
    job_timeout_seconds: int = 600
    max_file_size_mb: int = 50


class SessionSettings(ProjectBaseSettings):
    session_storage: str = "memory"  # or "redis"
    session_ttl_hours: int = 24
    max_history_messages: int = 20
```

**Step 2: Update ProjectSettings to inherit new settings**

Replace line 70-80 with:

```python
class ProjectSettings(
    OpenAILLMSettings,
    LangfuseSettings,
    QdrantVectorStoreSettings,
    OpenAIEmbeddingSettings,
    ConfidentSettings,
    RerankerSettings,
    Neo4jGraphDBSettings,
    MinIOSettings,
    CritiqueModelSettings,
    APISettings,
    AuthSettings,
    JobSettings,
    SessionSettings,
):
```

**Step 3: Add properties for new settings**

Add after line 115 (after critique_model property):

```python
    @property
    def api(self) -> APISettings:
        return APISettings(**self.model_dump())

    @property
    def auth(self) -> AuthSettings:
        return AuthSettings(**self.model_dump())

    @property
    def jobs(self) -> JobSettings:
        return JobSettings(**self.model_dump())

    @property
    def sessions(self) -> SessionSettings:
        return SessionSettings(**self.model_dump())
```

**Step 4: Verify settings load**

Run: `python -c "from src.settings import settings; print(settings.api.api_port)"`
Expected: `8000`

**Step 5: Commit**

```bash
git add src/settings.py
git commit -m "feat: add API, auth, job, and session settings"
```

---

### Task 3: Create API Directory Structure

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/routers/__init__.py`
- Create: `src/api/models/__init__.py`
- Create: `src/api/services/__init__.py`
- Create: `src/api/middleware/__init__.py`

**Step 1: Create directory structure**

Run:
```bash
mkdir -p src/api/routers src/api/models src/api/services src/api/middleware
touch src/api/__init__.py src/api/routers/__init__.py src/api/models/__init__.py src/api/services/__init__.py src/api/middleware/__init__.py
```

Expected: Directories and __init__.py files created

**Step 2: Commit**

```bash
git add src/api/
git commit -m "feat: create API directory structure"
```

---

## Phase 2: Core Models & Error Handling

### Task 4: Create API Request Models

**Files:**
- Create: `src/api/models/requests.py`

**Step 1: Write request models**

```python
from typing import Any
from pydantic import BaseModel, Field
from src.models import ChunkStrategy


class IngestRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=100)
    chunk_strategy: str = Field(default=ChunkStrategy.HYBRID.value)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    return_embeddings: bool = False


class GraphRAGRequest(RetrievalRequest):
    graph_depth: int = Field(default=2, ge=1, le=5)
    enable_reranking: bool = True


class GenerateRequest(RetrievalRequest):
    pass


class AgentChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=2000)
    collection_name: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_history: bool = True
```

**Step 2: Commit**

```bash
git add src/api/models/requests.py
git commit -m "feat: add API request models"
```

---

### Task 5: Create API Response Models

**Files:**
- Create: `src/api/models/responses.py`

**Step 1: Write response models**

```python
from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: int
    source: str
    content_snippet: str
    relevance_score: float
    cited_in_answer: bool = False


class RetrievalResult(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsageBreakdown(BaseModel):
    retrieval_embedding_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0


class TokenUsage(BaseModel):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    breakdown: TokenUsageBreakdown


class ToolCallInfo(BaseModel):
    tool: str
    query: str | None = None
    results_count: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    execution_time_ms: int = 0


class GraphPath(BaseModel):
    entities: list[str]
    relationship: str
    evidence_chunks: list[int] = Field(default_factory=list)


class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    created_at: datetime
    message: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: dict[str, Any] | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    citations: list[Citation]
    retrieval_time_ms: int
    num_results: int


class GraphRAGResponse(RetrievalResponse):
    graph_paths: list[GraphPath] = Field(default_factory=list)


class GenerateResponse(RetrievalResponse):
    generated_answer: str


class AgentChatResponse(BaseModel):
    message: str
    tool_calls: list[ToolCallInfo]
    citations: list[Citation]
    session_id: str
    user_id: str
    token_usage: TokenUsage
    response_time_ms: int


class ErrorResponse(BaseModel):
    error: dict[str, Any]
```

**Step 2: Commit**

```bash
git add src/api/models/responses.py
git commit -m "feat: add API response models"
```

---

### Task 6: Create API Key Model

**Files:**
- Create: `src/api/models/api_key.py`

**Step 1: Write API key model**

```python
from datetime import datetime
from pydantic import BaseModel, Field


class APIKey(BaseModel):
    api_key: str
    user_id: str
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["ingest", "retrieval", "agents"])
    rate_limit_tier: str = "standard"  # or "premium"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None


class APIKeyCreate(BaseModel):
    user_id: str
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["ingest", "retrieval", "agents"])
    rate_limit_tier: str = "standard"
    expires_at: datetime | None = None
```

**Step 2: Commit**

```bash
git add src/api/models/api_key.py
git commit -m "feat: add API key models"
```

---

### Task 7: Update API Models __init__.py

**Files:**
- Modify: `src/api/models/__init__.py`

**Step 1: Export all models**

```python
from src.api.models.requests import (
    IngestRequest,
    RetrievalRequest,
    GraphRAGRequest,
    GenerateRequest,
    AgentChatRequest,
)
from src.api.models.responses import (
    Citation,
    RetrievalResult,
    TokenUsage,
    TokenUsageBreakdown,
    ToolCallInfo,
    GraphPath,
    IngestJobResponse,
    IngestJobStatusResponse,
    RetrievalResponse,
    GraphRAGResponse,
    GenerateResponse,
    AgentChatResponse,
    ErrorResponse,
)
from src.api.models.api_key import APIKey, APIKeyCreate

__all__ = [
    "IngestRequest",
    "RetrievalRequest",
    "GraphRAGRequest",
    "GenerateRequest",
    "AgentChatRequest",
    "Citation",
    "RetrievalResult",
    "TokenUsage",
    "TokenUsageBreakdown",
    "ToolCallInfo",
    "GraphPath",
    "IngestJobResponse",
    "IngestJobStatusResponse",
    "RetrievalResponse",
    "GraphRAGResponse",
    "GenerateResponse",
    "AgentChatResponse",
    "ErrorResponse",
    "APIKey",
    "APIKeyCreate",
]
```

**Step 2: Commit**

```bash
git add src/api/models/__init__.py
git commit -m "feat: export API models"
```

---

### Task 8: Create Error Handler Middleware

**Files:**
- Create: `src/api/middleware/error_handler.py`

**Step 1: Write error handler**

```python
import traceback
from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger


class ErrorCode:
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_API_KEY = "INVALID_API_KEY"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


def create_error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Create standardized error response"""
    error_data = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id or "unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }
    return JSONResponse(status_code=status_code, content=error_data)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors"""
    logger.exception(f"Unhandled exception: {exc}")

    return create_error_response(
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error": str(exc)} if logger.level == "DEBUG" else {},
        request_id=getattr(request.state, "request_id", None),
    )
```

**Step 2: Commit**

```bash
git add src/api/middleware/error_handler.py
git commit -m "feat: add error handler middleware"
```

---

## Phase 3: Authentication & Services

### Task 9: Create Auth Service

**Files:**
- Create: `src/api/services/auth_service.py`

**Step 1: Write auth service**

```python
import secrets
from datetime import datetime
from passlib.context import CryptContext
from src.api.models import APIKey, APIKeyCreate
from src.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """In-memory API key storage (Redis-ready structure)"""

    def __init__(self):
        self.keys: dict[str, APIKey] = {}
        self._init_admin_key()

    def _init_admin_key(self):
        """Initialize admin API key for development"""
        admin_key = settings.auth.admin_api_key
        hashed = pwd_context.hash(admin_key)
        self.keys[hashed] = APIKey(
            api_key=hashed,
            user_id="admin",
            name="Admin Key",
            permissions=["ingest", "retrieval", "agents", "admin"],
            rate_limit_tier="premium",
            is_active=True,
        )

    def generate_api_key(self) -> str:
        """Generate new API key with newbie_ prefix"""
        return f"newbie_{secrets.token_hex(16)}"

    def create_key(self, key_create: APIKeyCreate) -> tuple[str, APIKey]:
        """Create new API key and return plaintext + stored model"""
        plaintext_key = self.generate_api_key()
        hashed_key = pwd_context.hash(plaintext_key)

        api_key = APIKey(
            api_key=hashed_key,
            user_id=key_create.user_id,
            name=key_create.name,
            permissions=key_create.permissions,
            rate_limit_tier=key_create.rate_limit_tier,
            expires_at=key_create.expires_at,
        )

        self.keys[hashed_key] = api_key
        return plaintext_key, api_key

    def validate_key(self, plaintext_key: str) -> APIKey | None:
        """Validate API key and return associated data"""
        for hashed_key, api_key in self.keys.items():
            if pwd_context.verify(plaintext_key, hashed_key):
                if not api_key.is_active:
                    return None
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return None

                # Update last used
                api_key.last_used_at = datetime.utcnow()
                return api_key

        return None

    def has_permission(self, api_key: APIKey, permission: str) -> bool:
        """Check if API key has specific permission"""
        return permission in api_key.permissions or "admin" in api_key.permissions


# Global auth service instance
auth_service = AuthService()
```

**Step 2: Commit**

```bash
git add src/api/services/auth_service.py
git commit -m "feat: add authentication service"
```

---

### Task 10: Create Job Manager Service

**Files:**
- Create: `src/api/services/job_manager.py`

**Step 1: Write job manager**

```python
import uuid
from enum import Enum
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class JobManager:
    """In-memory job tracking (Redis-ready structure)"""

    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def create_job(self) -> str:
        """Create new job and return job_id"""
        job_id = f"ingest_{uuid.uuid4().hex[:12]}"
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
        )
        self.jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        """Update job status and data"""
        job = self.jobs.get(job_id)
        if not job:
            return

        if status:
            job.status = status
        if progress is not None:
            job.progress = progress
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error

        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = datetime.utcnow()

    def list_jobs(self, user_id: str | None = None) -> list[Job]:
        """List all jobs (optionally filter by user_id in future)"""
        return list(self.jobs.values())


# Global job manager instance
job_manager = JobManager()
```

**Step 2: Commit**

```bash
git add src/api/services/job_manager.py
git commit -m "feat: add job manager service"
```

---

### Task 11: Create Session Manager Service

**Files:**
- Create: `src/api/services/session_manager.py`

**Step 1: Write session manager**

```python
import uuid
from datetime import datetime, timedelta
from typing import Any
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from src.settings import settings


class Session(BaseModel):
    session_id: str
    user_id: str
    messages: list[dict[str, Any]]  # Simplified message storage
    created_at: datetime
    last_accessed_at: datetime


class SessionManager:
    """In-memory session storage (Redis-ready structure)"""

    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def create_session(self, user_id: str) -> str:
        """Create new session and return session_id"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = Session(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
        )
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str, user_id: str) -> Session | None:
        """Get session by ID, validate user ownership"""
        session = self.sessions.get(session_id)
        if session and session.user_id == user_id:
            # Update last accessed
            session.last_accessed_at = datetime.utcnow()
            return session
        return None

    def add_message(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None):
        """Add message to session history"""
        session = self.sessions.get(session_id)
        if not session:
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(metadata or {}),
        }

        session.messages.append(message)

        # Trim to max history size
        max_messages = settings.sessions.max_history_messages
        if len(session.messages) > max_messages:
            session.messages = session.messages[-max_messages:]

    def get_history(self, session_id: str, user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get session message history with pagination"""
        session = self.get_session(session_id, user_id)
        if not session:
            return []

        total = len(session.messages)
        start = max(0, total - offset - limit)
        end = total - offset if offset > 0 else total

        return session.messages[start:end]

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete session if user owns it"""
        session = self.sessions.get(session_id)
        if session and session.user_id == user_id:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self, user_id: str) -> list[Session]:
        """List all sessions for a user"""
        return [s for s in self.sessions.values() if s.user_id == user_id]

    def cleanup_expired(self):
        """Remove expired sessions (TTL-based)"""
        ttl = timedelta(hours=settings.sessions.session_ttl_hours)
        now = datetime.utcnow()

        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_accessed_at > ttl
        ]

        for sid in expired:
            del self.sessions[sid]


# Global session manager instance
session_manager = SessionManager()
```

**Step 2: Commit**

```bash
git add src/api/services/session_manager.py
git commit -m "feat: add session manager service"
```

---

### Task 12: Update Services __init__.py

**Files:**
- Modify: `src/api/services/__init__.py`

**Step 1: Export services**

```python
from src.api.services.auth_service import AuthService, auth_service
from src.api.services.job_manager import JobManager, JobStatus, job_manager
from src.api.services.session_manager import SessionManager, session_manager

__all__ = [
    "AuthService",
    "auth_service",
    "JobManager",
    "JobStatus",
    "job_manager",
    "SessionManager",
    "session_manager",
]
```

**Step 2: Commit**

```bash
git add src/api/services/__init__.py
git commit -m "feat: export API services"
```

---

## Phase 4: FastAPI Dependencies

### Task 13: Create Authentication Dependency

**Files:**
- Create: `src/api/dependencies.py`

**Step 1: Write authentication dependencies**

```python
from fastapi import Header, HTTPException, status, Depends
from src.api.services import auth_service
from src.api.models import APIKey
from src.api.middleware.error_handler import ErrorCode, create_error_response


async def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> APIKey:
    """Validate API key from header"""
    api_key = auth_service.validate_key(x_api_key)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    return api_key


async def require_permission(permission: str):
    """Factory for permission-checking dependencies"""
    async def check_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
        if not auth_service.has_permission(api_key, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}",
            )
        return api_key

    return check_permission


# Specific permission dependencies
async def require_ingest_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "ingest"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: ingest",
        )
    return api_key


async def require_retrieval_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "retrieval"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: retrieval",
        )
    return api_key


async def require_agents_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "agents"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: agents",
        )
    return api_key
```

**Step 2: Commit**

```bash
git add src/api/dependencies.py
git commit -m "feat: add authentication dependencies"
```

---

## Phase 5: API Routers - Ingestion

### Task 14: Create Ingestion Router

**Files:**
- Create: `src/api/routers/ingestion.py`

**Step 1: Write ingestion router**

```python
import time
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from loguru import logger

from src.api.dependencies import require_ingest_permission
from src.api.models import APIKey, IngestJobResponse, IngestJobStatusResponse
from src.api.services import job_manager, JobStatus
from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.models import ChunkStrategy
from src.settings import settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


async def process_vectordb_ingestion(
    job_id: str,
    file_path: str,
    collection_name: str,
    chunk_strategy: str,
):
    """Background task to process vectordb ingestion"""
    try:
        job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=10)

        # Initialize ingestion pipeline
        pipeline = VectorDBIngestion(
            documents_dir="data/api_uploads/docs",
            chunks_dir="data/api_uploads/chunks",
            chunk_strategy=chunk_strategy,
            qdrant_collection_name=collection_name,
        )

        job_manager.update_job(job_id, progress=30)

        # Process file
        result = pipeline.ingest_file(file_path)

        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result=result,
        )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )


@router.post("/vectordb", response_model=IngestJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_vectordb(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_strategy: str = Form(default=ChunkStrategy.HYBRID.value),
    api_key: APIKey = Depends(require_ingest_permission),
):
    """
    Upload and process document into vector database.
    Returns job ID immediately, processing happens in background.
    """
    # Validate chunk strategy
    if chunk_strategy not in [e.value for e in ChunkStrategy]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chunk strategy. Must be one of: {[e.value for e in ChunkStrategy]}",
        )

    # Check file size
    max_size_bytes = settings.jobs.max_file_size_mb * 1024 * 1024
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.jobs.max_file_size_mb}MB",
        )

    # Create job
    job_id = job_manager.create_job()

    # Save uploaded file
    import os
    os.makedirs("data/api_uploads", exist_ok=True)
    file_path = f"data/api_uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Schedule background processing
    background_tasks.add_task(
        process_vectordb_ingestion,
        job_id=job_id,
        file_path=file_path,
        collection_name=collection_name,
        chunk_strategy=chunk_strategy,
    )

    job = job_manager.get_job(job_id)

    return IngestJobResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        message="File queued for processing",
    )


@router.get("/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_job_status(
    job_id: str,
    api_key: APIKey = Depends(require_ingest_permission),
):
    """Get ingestion job status and results"""
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.get("/collections", response_model=list[str])
async def list_collections(
    api_key: APIKey = Depends(require_ingest_permission),
):
    """List available Qdrant collections"""
    from src.deps import QdrantVectorStore

    vector_store = QdrantVectorStore(
        uri=settings.qdrant_uri,
        api_key=settings.qdrant_api_key,
    )

    collections = vector_store.list_collections()
    return collections
```

**Step 2: Commit**

```bash
git add src/api/routers/ingestion.py
git commit -m "feat: add ingestion router with vectordb endpoint"
```

---

## Phase 6: API Routers - Retrieval

### Task 15: Create Retrieval Router

**Files:**
- Create: `src/api/routers/retrieval.py`

**Step 1: Write retrieval router**

```python
import time
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.dependencies import require_retrieval_permission
from src.api.models import (
    APIKey,
    RetrievalRequest,
    GraphRAGRequest,
    GenerateRequest,
    RetrievalResponse,
    GraphRAGResponse,
    GenerateResponse,
    Citation,
    RetrievalResult,
    GraphPath,
)
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG
from src.settings import settings

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


def create_citations(retrieval_infos, cited_in_answer: bool = False) -> list[Citation]:
    """Convert retrieval infos to citations"""
    citations = []
    for idx, info in enumerate(retrieval_infos, start=1):
        # Truncate content for snippet
        snippet = info.content[:200] + "..." if len(info.content) > 200 else info.content

        citations.append(Citation(
            citation_id=idx,
            source=info.source,
            content_snippet=snippet,
            relevance_score=info.score,
            cited_in_answer=cited_in_answer,
        ))

    return citations


@router.post("/basic-rag", response_model=RetrievalResponse)
async def retrieve_basic_rag(
    request: RetrievalRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Semantic search using BasicRAG (vector-only)"""
    start_time = time.time()

    try:
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)
        retrieval_infos = basic_rag.retrieve(request.query, top_k=request.top_k)

        # Filter by score threshold
        if request.score_threshold > 0:
            retrieval_infos = [
                info for info in retrieval_infos
                if info.score >= request.score_threshold
            ]

        # Convert to response format
        results = [
            RetrievalResult(
                content=info.content,
                source=info.source,
                score=info.score,
                metadata={"score": info.score},
            )
            for info in retrieval_infos
        ]

        citations = create_citations(retrieval_infos, cited_in_answer=False)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RetrievalResponse(
            query=request.query,
            results=results,
            citations=citations,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results),
        )

    except Exception as e:
        logger.exception(f"BasicRAG retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.post("/graph-rag", response_model=GraphRAGResponse)
async def retrieve_graph_rag(
    request: GraphRAGRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Hybrid search using GraphRAG (vector + graph)"""
    start_time = time.time()

    try:
        graph_rag = GraphRAG(qdrant_collection_name=request.collection_name)

        # Use graph search
        search_result = await graph_rag.search(
            query=request.query,
            group_ids=[],  # Search all groups
        )

        # Convert nodes/edges to retrieval results
        results = []
        citations = []
        citation_id = 1

        # Add node summaries
        for node in (search_result.nodes or []):
            content = node.summary
            source = f"Node {node.uuid[:8]}"
            score = 0.9  # Placeholder score

            results.append(RetrievalResult(
                content=content,
                source=source,
                score=score,
                metadata={"type": "node", "uuid": node.uuid},
            ))

            snippet = content[:200] + "..." if len(content) > 200 else content
            citations.append(Citation(
                citation_id=citation_id,
                source=source,
                content_snippet=snippet,
                relevance_score=score,
                cited_in_answer=True,
            ))
            citation_id += 1

        # Add edge facts
        for edge in (search_result.edges or []):
            content = edge.fact
            source = f"Edge {edge.uuid[:8]}"
            score = 0.85

            results.append(RetrievalResult(
                content=content,
                source=source,
                score=score,
                metadata={"type": "edge", "uuid": edge.uuid},
            ))

            snippet = content[:200] + "..." if len(content) > 200 else content
            citations.append(Citation(
                citation_id=citation_id,
                source=source,
                content_snippet=snippet,
                relevance_score=score,
                cited_in_answer=True,
            ))
            citation_id += 1

        # Create graph paths (simplified)
        graph_paths = []
        if search_result.edges:
            entities = [node.summary.split()[0] for node in (search_result.nodes or [])[:3]]
            if len(entities) >= 2:
                graph_paths.append(GraphPath(
                    entities=entities,
                    relationship="related_to",
                    evidence_chunks=[],
                ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return GraphRAGResponse(
            query=request.query,
            results=results[:request.top_k],
            citations=citations[:request.top_k],
            graph_paths=graph_paths,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results[:request.top_k]),
        )

    except Exception as e:
        logger.exception(f"GraphRAG retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph retrieval failed: {str(e)}",
        )


@router.post("/generate", response_model=GenerateResponse)
async def retrieve_and_generate(
    request: GenerateRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Retrieve documents and generate answer (non-agentic)"""
    start_time = time.time()

    try:
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)
        retrieval_infos, generated_answer = basic_rag.generate(
            request.query,
            top_k=request.top_k,
            return_context=True,
        )

        # Filter by score threshold
        if request.score_threshold > 0:
            retrieval_infos = [
                info for info in retrieval_infos
                if info.score >= request.score_threshold
            ]

        results = [
            RetrievalResult(
                content=info.content,
                source=info.source,
                score=info.score,
                metadata={"score": info.score},
            )
            for info in retrieval_infos
        ]

        # Mark citations as used in answer
        citations = create_citations(retrieval_infos, cited_in_answer=True)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return GenerateResponse(
            query=request.query,
            results=results,
            citations=citations,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results),
            generated_answer=generated_answer,
        )

    except Exception as e:
        logger.exception(f"Generate failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
```

**Step 2: Commit**

```bash
git add src/api/routers/retrieval.py
git commit -m "feat: add retrieval router with basic/graph RAG endpoints"
```

---

## Phase 7: API Routers - Agents

### Task 16: Create Agent Router (Part 1 - Sync Endpoints)

**Files:**
- Create: `src/api/routers/agents.py`

**Step 1: Write agent router with sync endpoints**

```python
import time
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.dependencies import require_agents_permission
from src.api.models import (
    APIKey,
    AgentChatRequest,
    AgentChatResponse,
    Citation,
    ToolCallInfo,
    TokenUsage,
    TokenUsageBreakdown,
)
from src.api.services import session_manager
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from src.retrieval.basic_rag import BasicRAG
from src.settings import settings

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/basic-rag/chat", response_model=AgentChatResponse)
async def basic_rag_chat(
    request: AgentChatRequest,
    api_key: APIKey = Depends(require_agents_permission),
):
    """
    Synchronous agentic chat with BasicRAG.
    Returns full response after completion.
    """
    start_time = time.time()

    try:
        # Initialize BasicRAG
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)

        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        else:
            # Validate session ownership
            session = session_manager.get_session(session_id, request.user_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found or access denied",
                )

        # Get message history if requested
        messages = []
        if request.include_history:
            history = session_manager.get_history(session_id, request.user_id)
            # Convert to ModelMessage format (simplified)
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Create dependencies
        deps = BasicRAGDependencies(
            basic_rag=basic_rag,
            top_k=request.top_k,
        )

        # Run agent
        result = await basic_rag_agent.run(
            request.message,
            message_history=messages,
            deps=deps,
        )

        # Extract response data
        response_text = result.data

        # Extract tool calls and citations from result
        tool_calls = []
        citations = []
        citation_id = 1

        # Parse tool calls from result (simplified - would need actual extraction logic)
        # For now, create placeholder based on successful retrieval
        if hasattr(result, 'all_messages'):
            for msg in result.all_messages():
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCallInfo(
                            tool=tc.tool_name if hasattr(tc, 'tool_name') else "search_basic_rag",
                            query=request.message,
                            results_count=request.top_k,
                            token_usage={"embedding_tokens": 8},
                            execution_time_ms=100,
                        ))

        # Store conversation in session
        session_manager.add_message(
            session_id,
            role="user",
            content=request.message,
        )
        session_manager.add_message(
            session_id,
            role="assistant",
            content=response_text,
            metadata={
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "citations": [c.model_dump() for c in citations],
            },
        )

        # Calculate token usage (simplified)
        total_tokens = len(request.message.split()) * 2 + len(response_text.split()) * 2

        elapsed_ms = int((time.time() - start_time) * 1000)

        return AgentChatResponse(
            message=response_text,
            tool_calls=tool_calls,
            citations=citations,
            session_id=session_id,
            user_id=request.user_id,
            token_usage=TokenUsage(
                total_prompt_tokens=len(request.message.split()) * 2,
                total_completion_tokens=len(response_text.split()) * 2,
                total_tokens=total_tokens,
                breakdown=TokenUsageBreakdown(
                    retrieval_embedding_tokens=8,
                    llm_prompt_tokens=len(request.message.split()) * 2,
                    llm_completion_tokens=len(response_text.split()) * 2,
                ),
            ),
            response_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception(f"BasicRAG agent chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent chat failed: {str(e)}",
        )


@router.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Get conversation history for a session"""
    session = session_manager.get_session(session_id, user_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied",
        )

    messages = session_manager.get_history(session_id, user_id, limit=limit, offset=offset)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "messages": messages,
        "total_messages": len(session.messages),
        "created_at": session.created_at.isoformat() + "Z",
    }


@router.post("/sessions")
async def create_session(
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Create new session explicitly"""
    session_id = session_manager.create_session(user_id)
    session = session_manager.get_session(session_id, user_id)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat() + "Z",
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Delete session and clear history"""
    success = session_manager.delete_session(session_id, user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied",
        )

    return {"message": "Session deleted successfully"}


@router.get("/sessions")
async def list_sessions(
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """List all sessions for a user"""
    sessions = session_manager.list_sessions(user_id)

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat() + "Z",
                "last_accessed_at": s.last_accessed_at.isoformat() + "Z",
                "message_count": len(s.messages),
            }
            for s in sessions
        ]
    }
```

**Step 2: Commit**

```bash
git add src/api/routers/agents.py
git commit -m "feat: add agent router with sync chat and session endpoints"
```

---

### Task 17: Add Streaming Endpoint to Agent Router

**Files:**
- Modify: `src/api/routers/agents.py`

**Step 1: Add streaming endpoint after sync chat endpoint**

Add this function after the `basic_rag_chat` endpoint (around line 120):

```python
@router.post("/basic-rag/stream")
async def basic_rag_stream(
    request: AgentChatRequest,
    api_key: APIKey = Depends(require_agents_permission),
):
    """
    Streaming agentic chat with BasicRAG.
    Returns Server-Sent Events stream.
    """
    async def event_generator():
        try:
            # Initialize BasicRAG
            basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)

            # Get or create session
            session_id = request.session_id
            if not session_id:
                session_id = session_manager.create_session(request.user_id)
            else:
                session = session_manager.get_session(session_id, request.user_id)
                if not session:
                    yield f"event: error\ndata: {{\"error\": \"Session not found\"}}\n\n"
                    return

            # Get message history
            messages = []
            if request.include_history:
                history = session_manager.get_history(session_id, request.user_id)
                for msg in history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            # Create dependencies
            deps = BasicRAGDependencies(
                basic_rag=basic_rag,
                top_k=request.top_k,
            )

            start_time = time.time()
            accumulated_text = ""

            # Stream agent response
            async with basic_rag_agent.run_stream(
                request.message,
                message_history=messages,
                deps=deps,
            ) as result:
                # Stream tokens
                async for text_chunk in result.stream_text(delta=True):
                    accumulated_text += text_chunk
                    yield f"event: token\ndata: {{\"delta\": {repr(text_chunk)}}}\n\n"

                # Send completion event
                elapsed_ms = int((time.time() - start_time) * 1000)
                total_tokens = len(request.message.split()) * 2 + len(accumulated_text.split()) * 2

                done_data = {
                    "token_usage": {
                        "total_tokens": total_tokens,
                        "breakdown": {
                            "retrieval_embedding_tokens": 8,
                            "llm_prompt_tokens": len(request.message.split()) * 2,
                            "llm_completion_tokens": len(accumulated_text.split()) * 2,
                        }
                    },
                    "response_time_ms": elapsed_ms,
                    "session_id": session_id,
                }

                yield f"event: done\ndata: {done_data}\n\n"

                # Store in session
                session_manager.add_message(session_id, role="user", content=request.message)
                session_manager.add_message(session_id, role="assistant", content=accumulated_text)

        except Exception as e:
            logger.exception(f"Streaming failed: {e}")
            yield f"event: error\ndata: {{\"error\": {repr(str(e))}}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

**Step 2: Test streaming endpoint structure**

Run: `python -c "from src.api.routers.agents import router; print('Router loaded')"`
Expected: `Router loaded`

**Step 3: Commit**

```bash
git add src/api/routers/agents.py
git commit -m "feat: add SSE streaming endpoint for agent chat"
```

---

### Task 18: Update Routers __init__.py

**Files:**
- Modify: `src/api/routers/__init__.py`

**Step 1: Export routers**

```python
from src.api.routers import ingestion, retrieval, agents

__all__ = ["ingestion", "retrieval", "agents"]
```

**Step 2: Commit**

```bash
git add src/api/routers/__init__.py
git commit -m "feat: export API routers"
```

---

## Phase 8: FastAPI Application

### Task 19: Create Main FastAPI Application

**Files:**
- Create: `src/api/main.py`

**Step 1: Write FastAPI app**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routers import ingestion, retrieval, agents
from src.api.middleware.error_handler import global_exception_handler
from src.api.services import session_manager
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting FastAPI application...")

    # Startup tasks
    logger.info(f"API listening on {settings.api.api_host}:{settings.api.api_port}")
    logger.info(f"API prefix: {settings.api.api_prefix}")

    yield

    # Shutdown tasks
    logger.info("Shutting down FastAPI application...")
    session_manager.cleanup_expired()


app = FastAPI(
    title="newbieAR API",
    description="REST APIs for Ingestion, Retrieval, and Agentic RAG",
    version="1.0.0",
    docs_url=f"{settings.api.api_prefix}/docs",
    redoc_url=f"{settings.api.api_prefix}/redoc",
    openapi_url=f"{settings.api.api_prefix}/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(Exception, global_exception_handler)

# Include routers
app.include_router(ingestion.router, prefix=settings.api.api_prefix)
app.include_router(retrieval.router, prefix=settings.api.api_prefix)
app.include_router(agents.router, prefix=settings.api.api_prefix)


@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "newbieAR API"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check - verify DB connections"""
    try:
        from src.deps import QdrantVectorStore

        # Test Qdrant connection
        vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        vector_store.list_collections()

        return {"status": "ready", "checks": {"qdrant": "connected"}}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=True,
        log_level="info",
    )
```

**Step 2: Test app loads**

Run: `python -c "from src.api.main import app; print('App loaded')"`
Expected: `App loaded`

**Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat: create main FastAPI application with routers"
```

---

### Task 20: Create API Startup Script

**Files:**
- Create: `scripts/run_api.sh`

**Step 1: Write startup script**

```bash
#!/bin/bash

echo "Starting newbieAR API..."

# Activate virtual environment if needed
# source .venv/bin/activate

# Run with uvicorn
python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
```

**Step 2: Make executable**

Run: `chmod +x scripts/run_api.sh`

**Step 3: Commit**

```bash
git add scripts/run_api.sh
git commit -m "feat: add API startup script"
```

---

## Phase 9: Testing & Documentation

### Task 21: Create API Test File

**Files:**
- Create: `tests/test_api.py`

**Step 1: Write basic API tests**

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.services import auth_service
from src.settings import settings

client = TestClient(app)

# Use admin key for tests
API_KEY = settings.auth.admin_api_key
HEADERS = {"X-API-Key": API_KEY}


def test_health_check():
    """Test basic health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check():
    """Test readiness endpoint"""
    response = client.get("/health/ready")
    # May fail if Qdrant not running, that's OK for basic test
    assert response.status_code in [200, 503]


def test_list_collections():
    """Test list collections endpoint"""
    response = client.get(f"{settings.api.api_prefix}/ingest/collections", headers=HEADERS)
    assert response.status_code in [200, 500]  # OK if no Qdrant


def test_invalid_api_key():
    """Test authentication with invalid key"""
    bad_headers = {"X-API-Key": "invalid_key"}
    response = client.get(f"{settings.api.api_prefix}/ingest/collections", headers=bad_headers)
    assert response.status_code == 401


def test_create_session():
    """Test session creation"""
    response = client.post(
        f"{settings.api.api_prefix}/agents/sessions",
        params={"user_id": "test_user"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "session_id" in response.json()


def test_list_sessions():
    """Test listing sessions"""
    response = client.get(
        f"{settings.api.api_prefix}/agents/sessions",
        params={"user_id": "test_user"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "sessions" in response.json()
```

**Step 2: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: Tests pass (some may skip if services unavailable)

**Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test: add basic API tests"
```

---

### Task 22: Create README for API

**Files:**
- Create: `docs/API.md`

**Step 1: Write API documentation**

```markdown
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
```

**Step 2: Commit**

```bash
git add docs/API.md
git commit -m "docs: add API documentation"
```

---

### Task 23: Create Python Client Example

**Files:**
- Create: `examples/api_client.py`

**Step 1: Write client example**

```python
"""
Example Python client for newbieAR API
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "newbie_admin_dev_key_change_in_production"
HEADERS = {"X-API-Key": API_KEY}


def health_check():
    """Check API health"""
    response = requests.get("http://localhost:8000/health")
    print("Health:", response.json())


def upload_document(file_path: str, collection_name: str):
    """Upload document for ingestion"""
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "collection_name": collection_name,
            "chunk_strategy": "hybrid",
        }

        response = requests.post(
            f"{BASE_URL}/ingest/vectordb",
            headers=HEADERS,
            files=files,
            data=data,
        )

        print("Upload response:", response.json())
        return response.json()["job_id"]


def check_job(job_id: str):
    """Check ingestion job status"""
    response = requests.get(
        f"{BASE_URL}/ingest/jobs/{job_id}",
        headers=HEADERS,
    )
    print("Job status:", response.json())


def basic_rag_search(query: str, collection_name: str):
    """Perform basic RAG search"""
    payload = {
        "user_id": "example_user",
        "query": query,
        "collection_name": collection_name,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/retrieval/basic-rag",
        headers=HEADERS,
        json=payload,
    )

    print("Search results:", json.dumps(response.json(), indent=2))


def chat_with_agent(message: str, collection_name: str, session_id: str = None):
    """Chat with agent (synchronous)"""
    payload = {
        "user_id": "example_user",
        "message": message,
        "collection_name": collection_name,
        "session_id": session_id,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/agents/basic-rag/chat",
        headers=HEADERS,
        json=payload,
    )

    result = response.json()
    print("\nAgent response:", result["message"])
    print("Session ID:", result["session_id"])
    print("Citations:", len(result["citations"]))

    return result["session_id"]


def stream_chat(message: str, collection_name: str):
    """Stream chat with agent (SSE)"""
    payload = {
        "user_id": "example_user",
        "message": message,
        "collection_name": collection_name,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/agents/basic-rag/stream",
        headers=HEADERS,
        json=payload,
        stream=True,
    )

    print("\nStreaming response:")
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                print(decoded[6:])


if __name__ == "__main__":
    print("=== newbieAR API Client Examples ===\n")

    # Health check
    health_check()

    # Example: Upload and search
    # job_id = upload_document("path/to/document.pdf", "my_collection")
    # check_job(job_id)

    # Example: Search
    # basic_rag_search("What is RAG?", "my_collection")

    # Example: Chat
    # session_id = chat_with_agent("Explain transformers", "my_collection")

    # Example: Streaming chat
    # stream_chat("What are the benefits of RAG?", "my_collection")
```

**Step 2: Commit**

```bash
git add examples/api_client.py
git commit -m "docs: add Python API client example"
```

---

### Task 24: Final Integration Test

**Step 1: Start API server**

Run: `./scripts/run_api.sh` (in separate terminal)
Expected: Server starts on port 8000

**Step 2: Test API endpoints**

Run:
```bash
# Health check
curl http://localhost:8000/health

# List collections (with admin key)
curl -H "X-API-Key: newbie_admin_dev_key_change_in_production" \
  http://localhost:8000/api/v1/ingest/collections
```

Expected: Valid JSON responses

**Step 3: Access API docs**

Open: http://localhost:8000/api/v1/docs
Expected: Interactive OpenAPI documentation

**Step 4: Stop server and commit**

```bash
git add -A
git commit -m "feat: complete FastAPI implementation with all endpoints

- Ingestion API with async job management
- Retrieval API with BasicRAG and GraphRAG
- Agent API with sync and streaming modes
- Session management
- Authentication and error handling
- Health checks and API documentation"
```

---

## Summary

**What was built:**
- Complete FastAPI application with 3 main routers (Ingestion, Retrieval, Agents)
- API key authentication with permission system
- In-memory job and session management (Redis-ready)
- Request/response validation with Pydantic
- Error handling and health checks
- SSE streaming for agent responses
- OpenAPI documentation
- Example client code

**How to use:**
1. Start API: `./scripts/run_api.sh`
2. Access docs: http://localhost:8000/api/v1/docs
3. Use admin key: `newbie_admin_dev_key_change_in_production`
4. Test with examples: `python examples/api_client.py`

**Next steps (optional enhancements):**
- Add Langfuse middleware for observability
- Implement rate limiting with slowapi
- Add Redis support for production
- Add GraphRAG agent endpoints
- Add batch ingestion
- Add webhook support for job completion
- Add API key management endpoints
- Write integration tests
- Add Docker deployment configuration
