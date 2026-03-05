import asyncio
import tempfile
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from neo4j import AsyncGraphDatabase
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.ingestion.ingest_graphdb import GraphitiIngestion
from src.models import ChunkStrategy
from src.deps import QdrantVectorStore, GraphitiClient
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
        "chunk_count": result.get("chunk_count", 0),
        "chunks": result.get("chunks", []),
        "file_save_path": result.get("file_save_path"),
        "chunk_save_path": result.get("chunk_save_path"),
    }


@router.post("/graph")
async def ingest_graph(
    file: UploadFile = File(...),
    chunk_strategy: str = Form(ChunkStrategy.HIERARCHICAL.value),
):
    content = await file.read()
    original_filename = file.filename or "upload.pdf"

    suffix = os.path.splitext(original_filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ingestion = GraphitiIngestion(chunk_strategy=chunk_strategy)
        try:
            result = await ingestion.ingest_file(
                tmp_path, original_filename=original_filename
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            await ingestion.close()
    finally:
        os.unlink(tmp_path)

    return {
        "filename": result.get("filename", original_filename),
        "chunk_strategy": chunk_strategy,
        "chunk_count": result.get("chunk_count", 0),
        "chunks": result.get("chunks", []),
    }


async def get_neo4j_stats() -> dict:
    """Query Neo4j for node, relationship, and community counts."""
    driver = AsyncGraphDatabase.driver(
        settings.graph_db_uri,
        auth=(settings.graph_db_username, settings.graph_db_password),
    )
    try:
        async with driver.session() as session:
            nodes_result = await session.run("MATCH (n) RETURN count(n) AS nodes")
            nodes_record = await nodes_result.single()
            nodes = nodes_record["nodes"] if nodes_record else 0

            rels_result = await session.run(
                "MATCH ()-[r]->() RETURN count(r) AS relationships"
            )
            rels_record = await rels_result.single()
            relationships = rels_record["relationships"] if rels_record else 0

            comm_result = await session.run(
                "MATCH (n) WHERE n.group_id IS NOT NULL "
                "RETURN count(DISTINCT n.group_id) AS communities"
            )
            comm_record = await comm_result.single()
            communities = comm_record["communities"] if comm_record else 0
    finally:
        await driver.close()

    return {"nodes": nodes, "relationships": relationships, "communities": communities}


@router.get("/graph/summary")
async def get_graph_summary():
    try:
        stats = await get_neo4j_stats()
        return stats
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}")


@router.get("/collections/{name}")
async def get_collection_info(name: str):
    qs = QdrantVectorStore(uri=settings.qdrant_uri, api_key=settings.qdrant_api_key)
    info = qs.get_collection_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return info


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    qs = QdrantVectorStore(uri=settings.qdrant_uri, api_key=settings.qdrant_api_key)
    try:
        qs.delete_collection(name)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"deleted": name}


@router.post("/graph/clear")
async def clear_graph():
    client = GraphitiClient()
    try:
        await clear_data(client.driver)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear graph: {exc}")
    finally:
        await client.close()
    return {"status": "cleared"}
