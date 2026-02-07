import time
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.dependencies import require_retrieval_permission
from src.api.models import (
    APIKey,
    RetrievalRequest,
    GraphRAGRequest,
    GenerateRequest,
    RetrievalResponse,
    GraphRAGResponse,
    GenerateResponse,
    Citation,
    RetrievalResult,
    GraphPath,
)
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval
from src.settings import settings

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


def create_citations(retrieval_infos, cited_in_answer: bool = False) -> list[Citation]:
    """Convert retrieval infos to citations"""
    citations = []
    for idx, info in enumerate(retrieval_infos, start=1):
        # Truncate content for snippet
        snippet = info.content[:200] + "..." if len(info.content) > 200 else info.content

        citations.append(Citation(
            citation_id=idx,
            source=info.source,
            content_snippet=snippet,
            relevance_score=info.score,
            cited_in_answer=cited_in_answer,
        ))

    return citations


@router.post("/basic-rag", response_model=RetrievalResponse)
async def retrieve_basic_rag(
    request: RetrievalRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Semantic search using BasicRAG (vector-only)"""
    start_time = time.time()

    try:
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)
        retrieval_infos = basic_rag.retrieve(request.query, top_k=request.top_k)

        # Filter by score threshold
        if request.score_threshold > 0:
            retrieval_infos = [
                info for info in retrieval_infos
                if info.score >= request.score_threshold
            ]

        # Convert to response format
        results = [
            RetrievalResult(
                content=info.content,
                source=info.source,
                score=info.score,
                metadata={"score": info.score},
            )
            for info in retrieval_infos
        ]

        citations = create_citations(retrieval_infos, cited_in_answer=False)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RetrievalResponse(
            query=request.query,
            results=results,
            citations=citations,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results),
        )

    except Exception as e:
        logger.exception(f"BasicRAG retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.post("/graph-rag", response_model=GraphRAGResponse)
async def retrieve_graph_rag(
    request: GraphRAGRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Hybrid search using GraphRAG (vector + graph)"""
    start_time = time.time()

    try:
        graph_retrieval = GraphRetrieval()

        # Use graph retrieve method
        node_infos, edge_infos, episode_infos = await graph_retrieval.retrieve(
            query=request.query,
            num_results=request.top_k,
        )

        # Convert nodes/edges to retrieval results
        results = []
        citations = []
        citation_id = 1

        # Add node summaries
        for node in node_infos:
            content = node.summary
            source = f"Node {node.uuid[:8]}"
            score = 0.9  # Placeholder score

            results.append(RetrievalResult(
                content=content,
                source=source,
                score=score,
                metadata={"type": "node", "uuid": node.uuid},
            ))

            snippet = content[:200] + "..." if len(content) > 200 else content
            citations.append(Citation(
                citation_id=citation_id,
                source=source,
                content_snippet=snippet,
                relevance_score=score,
                cited_in_answer=True,
            ))
            citation_id += 1

        # Add edge facts
        for edge in edge_infos:
            content = edge.fact
            source = f"Edge {edge.uuid[:8]}"
            score = 0.85

            results.append(RetrievalResult(
                content=content,
                source=source,
                score=score,
                metadata={"type": "edge", "uuid": edge.uuid},
            ))

            snippet = content[:200] + "..." if len(content) > 200 else content
            citations.append(Citation(
                citation_id=citation_id,
                source=source,
                content_snippet=snippet,
                relevance_score=score,
                cited_in_answer=True,
            ))
            citation_id += 1

        # Create graph paths (simplified)
        graph_paths = []
        if edge_infos:
            entities = [node.summary.split()[0] for node in node_infos[:3]]
            if len(entities) >= 2:
                graph_paths.append(GraphPath(
                    entities=entities,
                    relationship="related_to",
                    evidence_chunks=[],
                ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        await graph_retrieval.close()

        return GraphRAGResponse(
            query=request.query,
            results=results[:request.top_k],
            citations=citations[:request.top_k],
            graph_paths=graph_paths,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results[:request.top_k]),
        )

    except Exception as e:
        logger.exception(f"GraphRAG retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph retrieval failed: {str(e)}",
        )


@router.post("/generate", response_model=GenerateResponse)
async def retrieve_and_generate(
    request: GenerateRequest,
    api_key: APIKey = Depends(require_retrieval_permission),
):
    """Retrieve documents and generate answer (non-agentic)"""
    start_time = time.time()

    try:
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)
        retrieval_infos, generated_answer = basic_rag.generate(
            request.query,
            top_k=request.top_k,
            return_context=True,
        )

        # Filter by score threshold
        if request.score_threshold > 0:
            retrieval_infos = [
                info for info in retrieval_infos
                if info.score >= request.score_threshold
            ]

        results = [
            RetrievalResult(
                content=info.content,
                source=info.source,
                score=info.score,
                metadata={"score": info.score},
            )
            for info in retrieval_infos
        ]

        # Mark citations as used in answer
        citations = create_citations(retrieval_infos, cited_in_answer=True)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return GenerateResponse(
            query=request.query,
            results=results,
            citations=citations,
            retrieval_time_ms=elapsed_ms,
            num_results=len(results),
            generated_answer=generated_answer,
        )

    except Exception as e:
        logger.exception(f"Generate failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
