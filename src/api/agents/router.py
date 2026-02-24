from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.api import deps
from src.api.agents.schemas import AgentRequest, GraphAgentRequest
from src.api.agents.service import AgentService
from src.api.agents.session_store import InMemorySessionStore

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/basic")
async def agent_basic(
    request: AgentRequest,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    service = AgentService(session_store)
    return StreamingResponse(
        service.stream_basic(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
    )


@router.post("/graph")
async def agent_graph(
    request: GraphAgentRequest,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    service = AgentService(session_store)
    return StreamingResponse(
        service.stream_graph(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        ),
        media_type="text/event-stream",
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_store: InMemorySessionStore = Depends(deps.get_session_store),
):
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}
