from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


class RetrievalService:
    def retrieve_basic(
        self, query: str, collection_name: str, top_k: int
    ) -> list[RetrievalInfo]:
        rag = BasicRAG(qdrant_collection_name=collection_name)
        return rag.retrieve(query, top_k=top_k)

    async def retrieve_graph(
        self, query: str, top_k: int
    ) -> tuple[
        list[GraphitiNodeInfo], list[GraphitiEdgeInfo], list[GraphitiEpisodeInfo]
    ]:
        retrieval = GraphRetrieval()
        try:
            nodes, edges, episodes = await retrieval.retrieve(query, num_results=top_k)
            return nodes, edges, episodes
        finally:
            await retrieval.close()
