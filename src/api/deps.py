from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.ingestion.job_store import InMemoryJobStore
    from src.api.agents.session_store import InMemorySessionStore

_job_store = None
_session_store = None


def get_job_store() -> "InMemoryJobStore":
    global _job_store
    if _job_store is None:
        from src.api.ingestion.job_store import InMemoryJobStore
        _job_store = InMemoryJobStore()
    return _job_store


def get_session_store() -> "InMemorySessionStore":
    global _session_store
    if _session_store is None:
        from src.api.agents.session_store import InMemorySessionStore
        _session_store = InMemorySessionStore()
    return _session_store
