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
