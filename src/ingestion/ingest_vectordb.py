from loguru import logger

from src.deps import QdrantVectorStore, OpenAIEmbeddingClient
from src.ingestion.load_document import DocLoader
from src.ingestion.chunk_document import DocChunker
from src.settings import settings


class VectorDBIngestion:
    def __init__(self, docs_dir: str, chunks_dir: str, collection_name: str = None):
        if collection_name is not None:
            settings.qdrant_collection_name = collection_name

        self.document_converter = DocLoader(output_dir=docs_dir)
        self.loader_and_chunker = DocChunker(output_dir=chunks_dir)
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

    def ingest_document(self, file_path: str) -> dict:
        """Ingest a single document into the vector database.
        Args:
            file_path: Path to input document
        Returns:
            Dictionary with all processing results
        """
        doc_save_path = self.document_converter.convert(file_path)
        logger.info(f"Document saved to {doc_save_path}")
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
            "doc_save_path": doc_save_path,
            "chunk_save_path": chunk_save_path,
            "vectordb_collection_name": settings.qdrant_collection_name,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--docs_dir", type=str, default="data/papers/docs")
    parser.add_argument("--chunks_dir", type=str, default="data/papers/chunks")
    parser.add_argument("--collection_name", type=str, required=False, default=None)
    args = parser.parse_args()

    pipeline = VectorDBIngestion(
        docs_dir=args.docs_dir,
        chunks_dir=args.chunks_dir,
        collection_name=args.collection_name,
    )
    results = pipeline.ingest_document(args.file_path)
    logger.info(f"Results: {results}")
