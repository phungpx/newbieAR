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
