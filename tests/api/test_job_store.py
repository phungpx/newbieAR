import pytest
from src.api.job_store import JobStatus, JobStore


def test_create_returns_unique_ids():
    store = JobStore()
    id1 = store.create()
    id2 = store.create()
    assert id1 != id2


def test_get_initial_state():
    store = JobStore()
    job_id = store.create()
    job = store.get(job_id)
    assert job is not None
    assert job["status"] == JobStatus.PENDING
    assert job["result"] is None
    assert job["error"] is None


def test_get_nonexistent_returns_none():
    store = JobStore()
    assert store.get("nonexistent-id") is None


def test_update_status_to_running():
    store = JobStore()
    job_id = store.create()
    store.update(job_id, status=JobStatus.RUNNING)
    job = store.get(job_id)
    assert job["status"] == JobStatus.RUNNING


def test_update_status_to_done_with_result():
    store = JobStore()
    job_id = store.create()
    result = {"goldens_count": 10, "output_dir": "data/goldens"}
    store.update(job_id, status=JobStatus.DONE, result=result)
    job = store.get(job_id)
    assert job["status"] == JobStatus.DONE
    assert job["result"] == result
    assert job["error"] is None


def test_update_status_to_failed_with_error():
    store = JobStore()
    job_id = store.create()
    store.update(job_id, status=JobStatus.FAILED, error="something went wrong")
    job = store.get(job_id)
    assert job["status"] == JobStatus.FAILED
    assert job["error"] == "something went wrong"
    assert job["result"] is None


def test_update_nonexistent_job_does_nothing():
    store = JobStore()
    # Should not raise
    store.update("nonexistent-id", status=JobStatus.RUNNING)


def test_multiple_jobs_isolated():
    store = JobStore()
    id1 = store.create()
    id2 = store.create()
    store.update(id1, status=JobStatus.DONE, result={"count": 5})
    job1 = store.get(id1)
    job2 = store.get(id2)
    assert job1["status"] == JobStatus.DONE
    assert job2["status"] == JobStatus.PENDING


def test_job_status_enum_values():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.DONE.value == "done"
    assert JobStatus.FAILED.value == "failed"
