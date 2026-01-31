import json
from pathlib import Path
from typing import List
from loguru import logger

from src.models.doc_info import DocInfo, DocStatus
from src.deps import QdrantVectorStore, OpenAIEmbeddingClient
from src.ingestion.load_document import DoclingDocumentConverter
from src.ingestion.chunk_document import DoclingChunker
from src.ingestion.config import MARKDOWN_DIR, CHUNKS_DIR
from src.settings import settings


class VectorDBIngestionPipeline:
    def __init__(
        self, markdown_dir: Path = MARKDOWN_DIR, chunks_dir: Path = CHUNKS_DIR
    ):
        self.markdown_converter = DoclingDocumentConverter(output_dir=markdown_dir)
        self.loader_and_chunker = DoclingChunker(output_dir=chunks_dir)
        self.embedding_client = OpenAIEmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model_id=settings.embedding_model,
        )
        self.vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        try:
            self.vector_store.create_collection(
                collection_name=settings.qdrant_collection_name,
                embedding_size=settings.embedding_dimensions,
                distance="cosine",
            )
            logger.info(
                f"Created collection {settings.qdrant_collection_name} with embedding size {settings.embedding_dimensions}"
            )
        except Exception as e:
            logger.error(
                f"Error creating collection {settings.qdrant_collection_name}: {e}"
            )
            raise e

    def ingest_document(self, file_path: str) -> DocInfo:
        """Ingest a single document into the vector database.
        Args:
            file_path: Path to input document
        Returns:
            Final DocInfo with all processing results
        """
        file_path = Path(file_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*60}")

        # Step 1: Convert to markdown
        doc_info = self.markdown_converter.convert(str(file_path))

        if doc_info.status == DocStatus.FAILED.value:
            logger.error(f"Pipeline failed at conversion step for {file_path.name}")
            return doc_info

        # Step 2: Chunk the document
        chunks, chunk_info = self.loader_and_chunker.chunk_document(str(file_path))

        # Merge chunk info into doc_info
        doc_info.chunk_path = chunk_info.chunk_path
        doc_info.chunk_count = chunk_info.chunk_count

        if chunk_info.status == DocStatus.FAILED.value:
            doc_info.status = DocStatus.FAILED.value
            doc_info.error = chunk_info.error
            logger.error(f"Pipeline failed at chunking step for {file_path.name}")
        else:
            doc_info.status = DocStatus.SUCCESS.value
            logger.info(f"\n✓ Pipeline completed successfully for {file_path.name}")
            logger.info(f"  Markdown: {doc_info.markdown_path}")
            logger.info(
                f"  Chunks: {doc_info.chunk_path} ({doc_info.chunk_count} chunks)"
            )

        # Step 3: Embed the document
        embeddings = []
        payloads = []
        for chunk in chunks:
            embeddings = self.embedding_client.embed_texts(chunk.text)
            payloads.append(chunk.model_dump())

        self.vector_store.add_embeddings(
            collection_name=settings.qdrant_collection_name,
            embeddings=embeddings,
            payloads=payloads,
        )

        return doc_info

    def ingest_documents(self, file_paths: List[str]) -> List[DocInfo]:
        """Ingest multiple documents into the vector database.
        Args:
            file_paths: List of file paths to process
        Returns:
            List of DocInfo objects
        """
        results = []

        logger.info(f"\nProcessing batch of {len(file_paths)} documents...")

        for file_path in file_paths:
            doc_info = self.ingest_document(file_path)
            results.append(doc_info)

        # Summary
        successful = sum(1 for r in results if r.status == DocStatus.SUCCESS.value)
        failed = sum(1 for r in results if r.status == DocStatus.FAILED.value)

        logger.info(f"\n{'='*60}")
        logger.info("Batch Processing Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(results)} | Success: {successful} | Failed: {failed}")

        return results


def ingest_document(file_path: str) -> dict:
    pipeline = VectorDBIngestionPipeline()
    doc_info = pipeline.ingest_document(file_path)
    return doc_info.model_dump()


def ingest_documents(file_paths: list[str]) -> list[dict]:
    pipeline = VectorDBIngestionPipeline()
    results = pipeline.ingest_documents(file_paths)
    return [doc.model_dump() for doc in results]


if __name__ == "__main__":
    # Test the full pipeline
    test_files = [
        "data/wikipedia/Albert_Einstein.pdf",
        "data/wikipedia/Isaac_Newton.txt",
        "data/wikipedia/Albert_Einstein.docx",
    ]

    pipeline = VectorDBIngestionPipeline()
    results = pipeline.ingest_documents(test_files)

    # Save summary
    output_file = Path("data/processing_summary.json")
    with output_file.open(mode="w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=4, ensure_ascii=False)

    logger.info(f"\nSummary saved to: {output_file}")
