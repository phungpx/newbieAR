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
