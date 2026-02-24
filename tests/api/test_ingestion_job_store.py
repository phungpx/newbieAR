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
