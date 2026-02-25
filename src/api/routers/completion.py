from loguru import logger
from fastapi import APIRouter, HTTPException, Request, status

from src.agents.agentic_rag import agentic_rag
from src.api.schemas import ChatRequest, CompletionResponse
from src.api.routers.sessions import store as session_store

router = APIRouter(prefix="/completion", tags=["completion"])


@router.post("", response_model=CompletionResponse)
async def completion(request: Request, body: ChatRequest) -> CompletionResponse:
    state = session_store.get(body.session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{body.session_id}' not found",
        )

    state.deps.clear_context()
    model = request.app.state.model

    try:
        result = await agentic_rag.run(
            body.message,
            model=model,
            message_history=state.messages,
            deps=state.deps,
        )
    except Exception as e:
        logger.exception(f"Completion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    state.messages.extend(result.all_messages())
    return CompletionResponse(
        text=result.data,
        contexts=state.deps.contexts or [],
        citations=state.deps.citations or [],
    )
