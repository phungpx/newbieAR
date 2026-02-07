from src.retrieval.basic_rag import BasicRAG
from src.ingestion.ingest_vectordb import VectorDBIngestion
from src.models import ChunkStrategy


class NewbieAR:
    def __init__(
        self,
        documents_dir: str,
        chunks_dir: str,
        chunk_strategy: str = ChunkStrategy.HYBRID.value,
        qdrant_collection_name: str = None,
    ):
        self.basic_rag = BasicRAG(
            qdrant_collection_name=qdrant_collection_name,
        )
        self.vector_db_ingestion = VectorDBIngestion(
            documents_dir=documents_dir,
            chunks_dir=chunks_dir,
            chunk_strategy=chunk_strategy,
            qdrant_collection_name=qdrant_collection_name,
        )

    def ingest_file(self, file_path: str):
        self.vector_db_ingestion.ingest_file(file_path)

    def ingest_files(self, file_paths: list[str]):
        for file_path in file_paths:
            self.ingest_file(file_path)

    def generate(self, query: str, top_k: int = 5, return_context: bool = False):
        return self.basic_rag.generate(
            query, top_k=top_k, return_context=return_context
        )

    def agent(self):
        pass

    def evaluate(self):
        pass

    def synthesize(self):
        pass
