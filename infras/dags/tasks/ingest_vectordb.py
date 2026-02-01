import os
import json
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from src.deps import MinIOClient, OpenAIEmbeddingClient, QdrantVectorStore
from src.ingestion.chunk_document import DocChunker
from src.ingestion.load_document import DocLoader
from src.settings import settings

BUCKET_UPLOADS = "uploads"
BUCKET_DOCS = "docs"
BUCKET_CHUNKS = "chunks"


def _get_minio_client() -> MinIOClient:
    return MinIOClient(
        endpoint=settings.minio.minio_endpoint,
        access_key=settings.minio.minio_access_key,
        secret_key=settings.minio.minio_secret_key,
        secure=settings.minio.minio_secure,
    )


# -------------------------------------------------------------------------
# TASK 1: Ingest Raw File
# -------------------------------------------------------------------------
def store_uploaded_file_to_minio_task(file_path: str, **kwargs: Any) -> dict[str, Any]:
    """
    Takes a local file path (from trigger), uploads to MinIO 'uploads' bucket.
    Returns the MinIO object path for downstream tasks.
    """
    try:
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = path_obj.stat().st_size
        logger.info(f"Storing file to MinIO: {path_obj.name} ({file_size} bytes)")

        minio_client = _get_minio_client()
        object_name = f"{BUCKET_UPLOADS}/{path_obj.name}"

        # Upload
        object_path = minio_client.upload_file(
            bucket_name=BUCKET_UPLOADS,
            object_name=object_name,
            file_path=path_obj,
        )

        logger.info(f"✓ Stored raw file: {object_path}")

        return {
            "status": "success",
            "object_path": object_path,  # e.g., "uploads/report.pdf"
            "bucket": BUCKET_UPLOADS,
            "file_name": path_obj.name,
            "file_size": file_size,
        }

    except Exception as e:
        logger.error(f"Failed to store file: {e}")
        return {"status": "failed", "error": str(e), "file_name": Path(file_path).name}


# -------------------------------------------------------------------------
# TASK 2: Chunk Document
# -------------------------------------------------------------------------
def chunk_document_task(
    minio_object_path: str,  # This should be the RAW file path from Task 1
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Downloads the RAW file from MinIO, chunks it, and uploads chunks to MinIO.

    Args:
        minio_object_path: Path to the raw file in MinIO (e.g., 'uploads/report.pdf')
    """
    temp_file = None
    try:
        if not minio_object_path:
            raise ValueError("No minio_object_path provided")

        logger.info(f"Preparing to chunk raw file: {minio_object_path}")
        minio_client = _get_minio_client()

        # Parse bucket and object name
        # Handle cases where path is "uploads/file.pdf" or just "file.pdf" inside implicit bucket
        if "/" in minio_object_path:
            bucket_name, object_name = minio_object_path.split("/", 1)
        else:
            bucket_name = BUCKET_UPLOADS
            object_name = minio_object_path

        # 1. Download Raw File to Temp
        # We preserve suffix (e.g. .pdf) so DocChunker knows how to handle it
        suffix = Path(object_name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()  # Close file handle so MinIO client can write to it

        minio_client.download_file(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=temp_path,
        )

        # 2. Chunk the Raw File
        logger.info(f"Running DocChunker on: {temp_path}")
        # Assuming DocChunker handles the raw parsing internally
        chunker = DocChunker(output_dir=None)
        chunks, _ = chunker.chunk_document(temp_path)

        if not chunks:
            logger.warning("DocChunker returned 0 chunks.")
            return {"status": "success", "chunk_count": 0, "chunks_path": None}

        # 3. Serialize and Upload Chunks to MinIO
        chunk_infos = [chunk.model_dump() for chunk in chunks]
        chunks_json = json.dumps(chunk_infos, indent=2, ensure_ascii=False)

        file_stem = Path(object_name).stem
        chunks_object_name = f"{BUCKET_CHUNKS}/{file_stem}_chunks.json"

        minio_client.upload_string(
            bucket_name=BUCKET_CHUNKS,
            object_name=chunks_object_name,
            content=chunks_json,
            content_type="application/json",
        )

        logger.info(
            f"✓ Successfully chunked {len(chunks)} parts to {chunks_object_name}"
        )

        return {
            "status": "success",
            "chunks_path": chunks_object_name,  # Pass this to embedding task
            "chunk_count": len(chunks),
            "original_file": object_name,
        }

    except Exception as e:
        error_msg = f"Failed to chunk document: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

    finally:
        # Cleanup temp raw file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


# -------------------------------------------------------------------------
# TASK 3: Embed & Load (Consumes Chunks JSON)
# -------------------------------------------------------------------------
def embed_and_store_task(
    chunks_object_path: str,  # Path to the JSON file in MinIO
    batch_size: int = 32,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Downloads chunks JSON, generates embeddings, and stores in Qdrant.
    """
    if not chunks_object_path:
        return {"status": "failed", "error": "Missing chunks_object_path"}

    try:
        minio_client = _get_minio_client()

        # 1. Load Chunks from MinIO (avoiding local temp file for JSON if possible)
        # Assuming we can read string/bytes directly, otherwise use temp file
        bucket, key = chunks_object_path.split("/", 1)

        # Download JSON content
        # Note: If your client doesn't have get_object/read, download to temp like before.
        # Here assuming a helper method or temp file approach:
        with tempfile.NamedTemporaryFile(delete=True) as tf:
            minio_client.download_file(
                bucket_name=bucket, object_name=key, file_path=tf.name
            )
            tf.seek(0)
            chunks_data = json.load(tf)

        if not chunks_data:
            return {"status": "skipped", "reason": "Empty chunks file"}

        # 2. Generate Embeddings
        embedding_client = OpenAIEmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model_id=settings.embedding_model,
        )

        logger.info(f"Generating embeddings for {len(chunks_data)} chunks...")
        chunk_texts = [c.get("text", "") for c in chunks_data]

        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            all_embeddings.extend(embedding_client.embed_texts(batch))

        # 3. Store in Qdrant
        logger.info(f"Storing {len(all_embeddings)} vectors to Qdrant...")
        vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )

        vector_store.create_collection(
            collection_name=settings.qdrant_collection_name,
            embedding_size=settings.embedding_dimensions,
            distance="cosine",
        )

        vector_store.add_embeddings(
            collection_name=settings.qdrant_collection_name,
            embeddings=all_embeddings,
            payloads=chunks_data,
        )

        return {
            "status": "success",
            "stored_count": len(all_embeddings),
            "collection": settings.qdrant_collection_name,
        }

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {"status": "failed", "error": str(e)}


# -------------------------------------------------------------------------
# OPTIONAL TASK: Convert to Markdown
# -------------------------------------------------------------------------
def convert_to_markdown_task(
    minio_object_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Optional: Converts raw file to markdown for display/debugging purposes.
    Does NOT pass data to chunking task anymore.
    """
    temp_file = None
    try:
        if not minio_object_path:
            return {"status": "skipped"}

        minio_client = _get_minio_client()
        bucket, object_name = minio_object_path.split("/", 1)

        # Download Raw
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(object_name).suffix
        )
        minio_client.download_file(
            bucket_name=bucket, object_name=object_name, file_path=temp_file.name
        )
        temp_file.close()

        # Convert
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DocLoader(output_dir=temp_dir)
            markdown_path = loader.convert(temp_file.name)

            # Upload Markdown
            md_filename = f"{Path(object_name).stem}.md"
            md_object_path = f"{BUCKET_DOCS}/{md_filename}"

            minio_client.upload_file(
                bucket_name=BUCKET_DOCS,
                object_name=md_object_path,
                file_path=Path(markdown_path),
            )

        return {"status": "success", "markdown_path": md_object_path}

    except Exception as e:
        logger.error(f"Markdown conversion failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
