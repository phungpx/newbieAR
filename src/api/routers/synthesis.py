import asyncio
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.job_store import JobStatus, job_store

router = APIRouter(prefix="/synthesis", tags=["synthesis"])


class SynthesisRequest(BaseModel):
    file_dir: str = "data/papers/files"
    output_dir: str = "data/goldens"
    topic: str = "paper"
    num_contexts: int = 5
    context_size: int = 5


@router.post("/upload")
async def upload_synthesis_files(files: list[UploadFile] = File(...)):
    tmp_dir = tempfile.mkdtemp(prefix="synthesis_")
    for file in files:
        content = await file.read()
        filename = file.filename or f"upload_{uuid4().hex}.pdf"
        dest = os.path.join(tmp_dir, filename)
        with open(dest, "wb") as f:
            f.write(content)
    return {"file_dir": tmp_dir, "file_count": len(files)}


def _run_synthesis(job_id: str, req: SynthesisRequest) -> None:
    # Lazy imports to avoid module-level side effects at app startup
    from src.synthesis.synthesize import (
        synthesizer,
        model,
        embedder,
        vector_store,
        STYLING_CONFIG,
    )
    from src.synthesis.generate_contexts import generate_contexts, save_goldens_to_files
    from src.settings import settings

    job_store.update(job_id, status=JobStatus.RUNNING)
    try:
        file_dir = Path(req.file_dir)
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = list(file_dir.glob("**/*.*"))
        goldens_count = 0

        for file_path in file_paths:
            contexts = asyncio.run(
                generate_contexts(
                    str(file_path),
                    model=model,
                    embedder=embedder,
                    vector_store=vector_store,
                    embedding_size=settings.embedding_dimensions,
                    num_contexts=req.num_contexts,
                    context_size=req.context_size,
                )
            )
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
                source_files=[str(file_path)] * len(contexts),
            )
            save_goldens_to_files(goldens, str(output_dir))
            goldens_count += len(goldens)

        job_store.update(
            job_id,
            status=JobStatus.DONE,
            result={"goldens_count": goldens_count, "output_dir": str(output_dir)},
        )
    except Exception as exc:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/jobs", status_code=202)
async def create_synthesis_job(
    req: SynthesisRequest, background_tasks: BackgroundTasks
):
    job_id = job_store.create()
    background_tasks.add_task(_run_synthesis, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
async def get_synthesis_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }
