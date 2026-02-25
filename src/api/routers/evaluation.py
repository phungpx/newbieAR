import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.api.job_store import JobStatus, job_store

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    goldens_dir: str = "data/goldens"
    collection_name: str = "research_papers"
    retrieval_window_size: int = 5
    threshold: float = 0.5
    force_rerun: bool = False


def _run_evaluation(job_id: str, req: EvaluationRequest) -> None:
    # Lazy imports to avoid module-level side effects at app startup
    from src.evaluation.evaluate import (
        create_metrics,
        create_llm_test_case,
        evaluate_llm_test_case_on_metrics,
    )

    job_store.update(job_id, status=JobStatus.RUNNING)
    try:
        metrics = create_metrics(threshold=req.threshold)
        goldens_dir = Path(req.goldens_dir)
        json_files = list(goldens_dir.glob("**/*.json"))

        evaluated = 0
        skipped = 0
        scores_accumulator: dict[str, list[float]] = {}

        for file_path in json_files:
            with open(file_path, encoding="utf-8") as f:
                sample = json.load(f)

            already_done = (
                sample.get("actual_output") is not None
                and sample.get("retrieval_contexts") is not None
                and sample.get("metrics") is not None
            )
            if already_done and not req.force_rerun:
                skipped += 1
                continue

            try:
                test_case, sample = create_llm_test_case(
                    file_path=str(file_path),
                    retrieval_window_size=req.retrieval_window_size,
                    collection_name=req.collection_name,
                )
                metrics_result = evaluate_llm_test_case_on_metrics(
                    test_case=test_case, metrics=metrics
                )
                sample["metrics"] = metrics_result
                with open(file_path, mode="w", encoding="utf-8") as f:
                    json.dump(sample, f, indent=4)

                for metric_name, metric_data in metrics_result.items():
                    scores_accumulator.setdefault(metric_name, []).append(
                        metric_data["score"]
                    )
                evaluated += 1
            except Exception:
                skipped += 1

        avg_scores = {
            name: sum(vals) / len(vals)
            for name, vals in scores_accumulator.items()
            if vals
        }

        job_store.update(
            job_id,
            status=JobStatus.DONE,
            result={"evaluated": evaluated, "skipped": skipped, "avg_scores": avg_scores},
        )
    except Exception as exc:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/jobs", status_code=202)
async def create_evaluation_job(req: EvaluationRequest, background_tasks: BackgroundTasks):
    job_id = job_store.create()
    background_tasks.add_task(_run_evaluation, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
async def get_evaluation_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }
