import asyncio
import tempfile
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.ingestion.ingest_graphdb import GraphitiIngestion
from src.models import ChunkStrategy
from src.deps import QdrantVectorStore
from src.settings import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/vector")
async def ingest_vector(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    chunk_strategy: str = Form(ChunkStrategy.HYBRID.value),
):
    content = await file.read()

    suffix = os.path.splitext(file.filename or "upload")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            ingestion = VectorDBIngestion(
                documents_dir="data/papers/docs",
                chunks_dir="data/papers/chunks",
                qdrant_collection_name=collection_name,
                chunk_strategy=chunk_strategy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            result = await asyncio.to_thread(ingestion.ingest_file, tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return {
        "collection_name": collection_name,
        "chunk_strategy": chunk_strategy,
        "file_save_path": result.get("file_save_path"),
        "chunk_save_path": result.get("chunk_save_path"),
    }


@router.post("/graph")
async def ingest_graph(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(ChunkStrategy.HIERARCHICAL.value),
):
    content = await file.read()

    suffix = os.path.splitext(file.filename or "upload")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ingestion = GraphitiIngestion(chunk_strategy=chunk_strategy)
        try:
            await ingestion.ingest_file(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            await ingestion.close()
    finally:
        os.unlink(tmp_path)

    return {
        "chunk_strategy": chunk_strategy,
        "filename": file.filename,
    }


@router.get("/collections/{name}")
async def get_collection_info(name: str):
    qs = QdrantVectorStore(uri=settings.qdrant_uri, api_key=settings.qdrant_api_key)
    info = qs.get_collection_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return info
