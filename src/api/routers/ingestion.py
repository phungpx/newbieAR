from loguru import logger
from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    BackgroundTasks,
)
from pathlib import Path

from src.api.dependencies import require_ingest_permission
from src.api.models import APIKey, IngestJobResponse, IngestJobStatusResponse
from src.api.services import job_manager, JobStatus
from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.models import ChunkStrategy
from src.settings import settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

FILE_STORAGE_DIR = Path("data/api_uploads")


async def process_vectordb_ingestion(
    job_id: str,
    file_path: str,
    qdrant_collection_name: str,
    chunk_strategy: str,
):
    try:
        job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=10)

        pipeline = VectorDBIngestion(
            documents_dir=f"{FILE_STORAGE_DIR}/docs",
            chunks_dir=f"{FILE_STORAGE_DIR}/chunks",
            chunk_strategy=chunk_strategy,
            qdrant_collection_name=qdrant_collection_name,
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


@router.post(
    "/ingest_file",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
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
    if not FILE_STORAGE_DIR.exists():
        FILE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    file_path = Path(FILE_STORAGE_DIR) / file.filename

    with file_path.open(mode="wb") as f:
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
    from src.deps import QdrantVectorStore

    qdrant_client = QdrantVectorStore(
        uri=settings.qdrant_uri,
        api_key=settings.qdrant_api_key,
    )

    collections = qdrant_client.list_collections()
    return collections
