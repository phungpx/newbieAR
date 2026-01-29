from pathlib import Path
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.settings import settings
from src.deps import (
    QdrantVectorStore,
    DocumentLoader,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
)


class Ingestion:
    def __init__(
        self,
        embedding_provider: str = "openai",
        chunk_size: int = 512,
        chunk_overlap: int = 20,
    ):
        if embedding_provider == "openai":
            self.embedding_client = OpenAIEmbedding(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                model_id=settings.embedding_model_id,
            )
        elif embedding_provider == "sentence-transformer":
            self.embedding_client = SentenceTransformerEmbedding(
                model_name=settings.embedding_model_name,
                batch_size=settings.embedding_batch_size,
                model_dim=settings.embedding_dimensions,
            )

        self.vector_store = QdrantVectorStore(
            uri=settings.qdrant_url,
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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def ingest_documents(self, file_paths: list[str]):
        chunks: list[Document] = []
        for file_path in file_paths:
            documents = DocumentLoader(file_path).load()
            for chunk in self.text_splitter.split_documents(documents):
                chunk.metadata["source"] = file_path
                chunks.append(chunk)

        logger.info(f"Chunked {len(chunks)} documents")

        embeddings = []
        for chunk in chunks:
            embedding = self.embedding_client.embed([chunk.page_content])
            embeddings.extend(embedding)
            logger.info(f"Embedded {chunk.page_content}")

        logger.info(f"Embedded {len(embeddings)} documents")

        payloads = [
            {"content": chunk.page_content, "sources": chunk.metadata["source"]}
            for chunk in chunks
        ]
        self.vector_store.add_embeddings(
            collection_name=settings.qdrant_collection_name,
            embeddings=embeddings,
            payloads=payloads,
            batch_size=2,
        )
        logger.info(
            f"Added {len(embeddings)} documents to collection {settings.qdrant_collection_name}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, required=True)
    parser.add_argument("--collection_name", type=str, required=False)
    args = parser.parse_args()

    file_dir = Path(args.file_dir)

    if not file_dir.exists():
        raise ValueError(f"File directory {file_dir} does not exist")
    if not file_dir.is_dir():
        raise ValueError(f"File directory {file_dir} is not a directory")

    file_paths = [str(file) for file in file_dir.glob("**/*.*")]

    if args.collection_name:
        settings.qdrant_collection_name = args.collection_name

    ingestion = Ingestion(embedding_provider="openai")
    ingestion.ingest_documents(file_paths)
