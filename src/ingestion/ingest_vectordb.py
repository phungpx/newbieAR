from loguru import logger
from src.deps import QdrantVectorStore, OpenAIEmbedding
from src.deps import DocumentChunker, DocumentLoader
from src.settings import settings
from src.models import ChunkStrategy


class VectorDBIngestion:
    def __init__(
        self,
        documents_dir: str,
        chunks_dir: str,
        chunk_strategy: str = ChunkStrategy.HYBRID.value,
        qdrant_collection_name: str = None,
    ):
        if chunk_strategy not in [e.value for e in ChunkStrategy]:
            raise ValueError(
                f"Invalid chunk strategy '{chunk_strategy}'. Must be one of: {ChunkStrategy.values()}"
            )
        self.chunk_strategy = chunk_strategy
        if qdrant_collection_name is not None:
            settings.qdrant_collection_name = qdrant_collection_name

        self.document_converter = DocumentLoader(output_dir=documents_dir)
        self.loader_and_chunker = DocumentChunker(
            output_dir=chunks_dir,
            strategy=chunk_strategy,
        )
        self.embedding_client = OpenAIEmbedding(
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

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file into the vector database.
        Args:
            file_path: Path to input file
        Returns:
            Dictionary with all processing results
        """
        file_save_path = self.document_converter.convert(file_path)
        logger.info(f"File saved to {file_save_path}")
        chunks, chunk_save_path = self.loader_and_chunker.chunk_document(file_path)

        embeddings, payloads = [], []
        for chunk in chunks:
            embeddings.extend(self.embedding_client.embed_texts(chunk.text))
            payloads.append(chunk.model_dump())

        self.vector_store.add_embeddings(
            collection_name=settings.qdrant_collection_name,
            embeddings=embeddings,
            payloads=payloads,
        )
        logger.info(
            f"{len(embeddings)} embeddings and {len(payloads)} payloads added to vector store: {settings.qdrant_collection_name}"
        )

        return {
            "file_save_path": file_save_path,
            "chunk_save_path": chunk_save_path,
            "qdrant_collection_name": settings.qdrant_collection_name,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text_tokens": c.text_tokens,
                    "text_preview": c.text[:120],
                }
                for c in chunks
            ],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--documents_dir", type=str, default="data/papers/docs")
    parser.add_argument("--chunks_dir", type=str, default="data/papers/chunks")
    parser.add_argument("--collection_name", type=str, required=False, default=None)
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default=ChunkStrategy.HYBRID.value,
        choices=[e.value for e in ChunkStrategy],
        help="Chunking strategy to use (default: hybrid)",
    )
    args = parser.parse_args()

    pipeline = VectorDBIngestion(
        documents_dir=args.documents_dir,
        chunks_dir=args.chunks_dir,
        qdrant_collection_name=args.qdrant_collection_name,
        chunk_strategy=args.chunk_strategy,
    )
    results = pipeline.ingest_file(args.file_path)
    logger.info(f"Results: {results}")
