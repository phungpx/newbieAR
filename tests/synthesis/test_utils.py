import pytest
from unittest.mock import MagicMock, patch

from src.synthesis.utils import build_contexts_from_doc


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_point(chunk_idx: int, text: str):
    point = MagicMock()
    point.payload = {"text": text, "chunk_idx": chunk_idx}
    return point


def _make_query_result(*points):
    result = MagicMock()
    result.points = list(points)
    return result


def _patch_chunker(texts: list[str]):
    """Patch DocumentChunker so chunk_document() returns chunks with given texts."""
    chunks = [MagicMock(text=t) for t in texts]
    mock_instance = MagicMock()
    mock_instance.chunk_document.return_value = (chunks, None)
    return patch("src.synthesis.utils.DocumentChunker", return_value=mock_instance)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def embedder():
    m = MagicMock()
    m.embed_texts.return_value = [[float(i), 0.0] for i in range(10)]
    return m


@pytest.fixture
def vector_store():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_document_returns_empty_list(embedder, vector_store):
    with _patch_chunker([]):
        result = build_contexts_from_doc(
            "/fake/doc.pdf", embedder, vector_store, embedding_size=2
        )
    assert result == []
    embedder.embed_texts.assert_not_called()
    vector_store.create_collection.assert_not_called()
    vector_store.delete_collection.assert_not_called()


def test_normal_case_returns_num_contexts_contexts(embedder, vector_store):
    texts = [f"para {i}" for i in range(5)]
    embedder.embed_texts.return_value = [[float(i), 0.0] for i in range(5)]
    # query returns 4 points; seed_idx=1 should be excluded from neighbors
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "para 0"),
        _make_point(1, "para 1"),
        _make_point(2, "para 2"),
        _make_point(3, "para 3"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=1):
            result = build_contexts_from_doc(
                "/fake/doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=3, context_size=3,
            )
    assert len(result) == 3
    # seed_idx=1 → neighbors=[chunk_idx 0, chunk_idx 2] → context length = 3
    assert result[0][0] == "para 1"
    assert len(result[0]) == 3


def test_seed_is_excluded_from_neighbors(embedder, vector_store):
    texts = ["s0", "s1", "s2", "s3", "s4"]
    embedder.embed_texts.return_value = [[float(i), 0.0] for i in range(5)]
    # query returns seed (idx=2) as first result — it must be filtered out
    vector_store.query.return_value = _make_query_result(
        _make_point(2, "s2"),  # ← seed, must be excluded
        _make_point(0, "s0"),
        _make_point(4, "s4"),
        _make_point(3, "s3"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=2):
            result = build_contexts_from_doc(
                "/fake/doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=1, context_size=3,
            )
    context = result[0]
    assert context[0] == "s2"       # seed is first
    assert "s2" not in context[1:]  # seed NOT repeated as a neighbor


def test_collection_deleted_on_success(embedder, vector_store):
    texts = ["p0", "p1", "p2"]
    embedder.embed_texts.return_value = [[0.0, 0.0]] * 3
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "p0"), _make_point(1, "p1"), _make_point(2, "p2"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=0):
            build_contexts_from_doc(
                "/fake/my_doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=1,
            )
    vector_store.delete_collection.assert_called_once_with("synthesis_my_doc")


def test_collection_deleted_on_error(embedder, vector_store):
    texts = ["p0", "p1"]
    embedder.embed_texts.return_value = [[0.0, 0.0]] * 2
    vector_store.query.side_effect = RuntimeError("Qdrant down")
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=0):
            with pytest.raises(RuntimeError, match="Qdrant down"):
                build_contexts_from_doc(
                    "/fake/err_doc.pdf", embedder, vector_store,
                    embedding_size=2, num_contexts=1,
                )
    # delete_collection must still be called even after the error
    vector_store.delete_collection.assert_called_once_with("synthesis_err_doc")


def test_uses_hierarchical_chunker_strategy(embedder, vector_store):
    with patch("src.synthesis.utils.DocumentChunker") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.chunk_document.return_value = ([], None)
        mock_cls.return_value = mock_instance
        build_contexts_from_doc("/fake/doc.pdf", embedder, vector_store, embedding_size=2)
        mock_cls.assert_called_once_with(strategy="hierarchical")


def test_num_contexts_capped_by_chunk_count(embedder, vector_store):
    """num_contexts > N should produce only N contexts."""
    texts = ["only", "two"]
    embedder.embed_texts.return_value = [[0.0, 0.0], [1.0, 0.0]]
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "only"), _make_point(1, "two"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", side_effect=[0, 1]):
            result = build_contexts_from_doc(
                "/fake/small.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=10, context_size=3,
            )
    assert len(result) == 2  # capped at min(10, 2) = 2
