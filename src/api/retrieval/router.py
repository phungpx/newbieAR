from fastapi import APIRouter
from src.api.retrieval.schemas import (
    BasicRetrievalRequest,
    BasicRetrievalResponse,
    GraphRetrievalRequest,
    GraphRetrievalResponse,
)
from src.api.retrieval.service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/basic", response_model=BasicRetrievalResponse)
async def retrieve_basic(request: BasicRetrievalRequest):
    service = RetrievalService()
    results = service.retrieve_basic(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )
    return BasicRetrievalResponse(results=results)


@router.post("/graph", response_model=GraphRetrievalResponse)
async def retrieve_graph(request: GraphRetrievalRequest):
    service = RetrievalService()
    nodes, edges, episodes = await service.retrieve_graph(
        query=request.query, top_k=request.top_k
    )
    return GraphRetrievalResponse(nodes=nodes, edges=edges, episodes=episodes)
