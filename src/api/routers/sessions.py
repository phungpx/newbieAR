from fastapi import APIRouter, HTTPException, status

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRAG
from src.agents.deps import AgentDependencies
from src.api.schemas import CreateSessionRequest, CreateSessionResponse, DeleteSessionResponse
from src.api.session_store import SessionStore

router = APIRouter(prefix="/sessions", tags=["sessions"])
store = SessionStore()


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    deps = AgentDependencies(
        basic_rag=BasicRAG(qdrant_collection_name=request.collection_name),
        graph_rag=GraphRAG(),
        top_k=request.top_k,
    )
    session_id = store.create(
        deps=deps,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )
    return CreateSessionResponse(
        session_id=session_id,
        collection_name=request.collection_name,
        top_k=request.top_k,
    )


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    if store.get(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    store.delete(session_id)
    return DeleteSessionResponse(message="Session deleted")
