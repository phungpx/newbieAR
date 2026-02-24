import asyncio
import os
import tempfile
from src.api.ingestion.job_store import InMemoryJobStore, JobStatus
from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.ingestion.ingest_graphdb import GraphitiIngestion
from src.models import ChunkStrategy


class IngestionService:
    def __init__(self, job_store: InMemoryJobStore):
        self.job_store = job_store

    async def ingest_vectordb(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        chunk_strategy: str,
    ) -> str:
        job = self.job_store.create_job()
        asyncio.create_task(
            self._run_vectordb(job.job_id, file_bytes, filename, collection_name, chunk_strategy)
        )
        return job.job_id

    async def _run_vectordb(
        self,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        chunk_strategy: str,
    ):
        self.job_store.update_job(job_id, JobStatus.RUNNING)
        tmp_path = None
        try:
            suffix = os.path.splitext(filename)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            ingestion = VectorDBIngestion(
                documents_dir="data/papers/docs",
                chunks_dir="data/papers/chunks",
                chunk_strategy=chunk_strategy,
                qdrant_collection_name=collection_name,
            )
            result = ingestion.ingest_file(tmp_path)
            self.job_store.update_job(job_id, JobStatus.DONE, result=result)
        except Exception as e:
            self.job_store.update_job(job_id, JobStatus.FAILED, error=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def ingest_graphdb(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_strategy: str,
    ) -> str:
        job = self.job_store.create_job()
        asyncio.create_task(
            self._run_graphdb(job.job_id, file_bytes, filename, chunk_strategy)
        )
        return job.job_id

    async def _run_graphdb(
        self,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        chunk_strategy: str,
    ):
        self.job_store.update_job(job_id, JobStatus.RUNNING)
        tmp_path = None
        ingestion = None
        try:
            suffix = os.path.splitext(filename)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            ingestion = GraphitiIngestion(chunk_strategy=chunk_strategy)
            await ingestion.ingest_file(tmp_path)
            self.job_store.update_job(job_id, JobStatus.DONE, result={"filename": filename})
        except Exception as e:
            self.job_store.update_job(job_id, JobStatus.FAILED, error=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if ingestion:
                await ingestion.close()
