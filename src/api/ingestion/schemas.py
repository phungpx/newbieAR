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
