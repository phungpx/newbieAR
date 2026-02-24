from unittest.mock import MagicMock, patch
from src.settings import settings


def test_basic_rag_does_not_mutate_global_settings():
    original = settings.qdrant_collection_name

    with patch("src.retrieval.basic_rag.QdrantVectorStore"), \
         patch("src.retrieval.basic_rag.OpenAIEmbedding"), \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):
        from src.retrieval.basic_rag import BasicRAG
        BasicRAG(qdrant_collection_name="test_collection")

    assert settings.qdrant_collection_name == original, (
        "BasicRAG.__init__ must not mutate the global settings singleton"
    )


def test_basic_rag_uses_provided_collection_name():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding"), \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):
        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="my_collection")

    assert rag.collection_name == "my_collection"
