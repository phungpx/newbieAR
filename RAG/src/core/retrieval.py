from src.deps.vector_stores import QdrantVectorStore
from src.deps.embeddings import EmbeddingClient
from src.settings import settings

vector_store = QdrantVectorStore(
    uri=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

embedding_client = EmbeddingClient(
    model_name=settings.embedding_model_name,
    batch_size=settings.embedding_batch_size,
    model_dim=settings.embedding_dimensions,
)


def retrieve(query: str) -> list[str]:
    embedding = embedding_client.embed([query])
    retrieved_documents = vector_store.query(
        collection_name=settings.qdrant_collection_name,
        query_vector=embedding,
    )
    return [document.payload for document in retrieved_documents]


if __name__ == "__main__":
    print(retrieve("What is the capital of France?"))
