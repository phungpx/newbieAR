import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.api.retrieval.service import RetrievalService
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


@pytest.fixture
def service():
    return RetrievalService()


def test_retrieve_basic_returns_retrieval_infos(service):
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        RetrievalInfo(content="doc text", source="file.pdf - Chunk #1", score=0.9)
    ]
    with patch("src.api.retrieval.service.BasicRAG", return_value=mock_rag):
        results = service.retrieve_basic(query="test", collection_name="col", top_k=3)
    assert len(results) == 1
    assert results[0].content == "doc text"
    mock_rag.retrieve.assert_called_once_with("test", top_k=3)


@pytest.mark.asyncio
async def test_retrieve_graph_returns_node_edge_episode(service):
    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve.return_value = (
        [GraphitiNodeInfo(uuid="n1", summary="Node 1")],
        [GraphitiEdgeInfo(uuid="e1", fact="Fact 1")],
        [GraphitiEpisodeInfo(uuid="ep1", content="Episode 1")],
    )
    mock_retrieval.close = AsyncMock()
    with patch("src.api.retrieval.service.GraphRetrieval", return_value=mock_retrieval):
        nodes, edges, episodes = await service.retrieve_graph(query="test", top_k=5)
    assert len(nodes) == 1
    assert nodes[0].uuid == "n1"
    assert len(edges) == 1
    assert len(episodes) == 1
    mock_retrieval.close.assert_awaited_once()
