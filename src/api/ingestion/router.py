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
