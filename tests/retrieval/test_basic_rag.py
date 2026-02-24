import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.settings import settings
from src.models import RetrievalInfo


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


async def test_retrieve_filters_below_score_threshold():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point_low = MagicMock()
        mock_point_low.payload = {"text": "low score doc", "filename": "a.txt", "chunk_id": "1"}
        mock_point_low.score = 0.2

        mock_point_high = MagicMock()
        mock_point_high.payload = {"text": "high score doc", "filename": "b.txt", "chunk_id": "2"}
        mock_point_high.score = 0.8

        mock_result = MagicMock()
        mock_result.points = [mock_point_low, mock_point_high]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        results = await rag.retrieve("test query", top_k=5, score_threshold=0.5)

    assert len(results) == 1
    assert results[0].content == "high score doc"


async def test_retrieve_reranks_when_cross_encoder_set():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point_a = MagicMock()
        mock_point_a.payload = {"text": "doc A", "filename": "a.txt", "chunk_id": "1"}
        mock_point_a.score = 0.9  # high vector score

        mock_point_b = MagicMock()
        mock_point_b.payload = {"text": "doc B", "filename": "b.txt", "chunk_id": "2"}
        mock_point_b.score = 0.5  # lower vector score

        mock_result = MagicMock()
        mock_result.points = [mock_point_a, mock_point_b]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        # Cross-encoder ranks doc B higher than doc A
        mock_cross_encoder = AsyncMock()
        mock_cross_encoder.rank.return_value = [("doc B", 0.95), ("doc A", 0.3)]
        rag.cross_encoder = mock_cross_encoder

        results = await rag.retrieve("test query", top_k=5)

    assert results[0].content == "doc B"
    assert results[1].content == "doc A"
    assert results[0].score == pytest.approx(0.95)
    assert results[1].score == pytest.approx(0.3)


async def test_retrieve_returns_all_when_threshold_is_zero():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc", "filename": "a.txt", "chunk_id": "1"}
        mock_point.score = 0.1

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        results = await rag.retrieve("test query", top_k=5, score_threshold=0.0)

    assert len(results) == 1
