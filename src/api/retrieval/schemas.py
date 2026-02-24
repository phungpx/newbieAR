from pydantic import BaseModel
from src.models import RetrievalInfo, GraphitiNodeInfo, GraphitiEdgeInfo, GraphitiEpisodeInfo


class BasicRetrievalRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5


class BasicRetrievalResponse(BaseModel):
    results: list[RetrievalInfo]


class GraphRetrievalRequest(BaseModel):
    query: str
    top_k: int = 5


class GraphRetrievalResponse(BaseModel):
    nodes: list[GraphitiNodeInfo]
    edges: list[GraphitiEdgeInfo]
    episodes: list[GraphitiEpisodeInfo]
