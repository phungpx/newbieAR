import uuid
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
        }
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: JobStatus | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error


# Module-level singleton shared across routers
job_store = JobStore()
