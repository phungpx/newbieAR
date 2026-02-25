from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG
from src.deps.cross_encoder.sentence_transformers_reranker import SentenceTransformersReranker

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


class VectorRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_name: str
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = False


class GraphRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 10


@router.post("/vector")
async def retrieve_vector(req: VectorRetrieveRequest):
    basic_rag = BasicRAG(qdrant_collection_name=req.collection_name)
    if req.rerank:
        basic_rag.cross_encoder = SentenceTransformersReranker()

    try:
        results = await basic_rag.retrieve(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "results": [
            {"content": r.content, "source": r.source, "score": r.score}
            for r in results
        ]
    }


@router.post("/graph")
async def retrieve_graph(req: GraphRetrieveRequest):
    graph_rag = GraphRAG()
    try:
        contexts, citations = await graph_rag.retrieve(query=req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await graph_rag.close()

    return {"contexts": contexts, "citations": citations}
