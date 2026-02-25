import json
from loguru import logger
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.agents.agentic_rag import agentic_rag
from src.api.schemas import ChatRequest
from src.api.routers.sessions import store as session_store

router = APIRouter(prefix="/chat/stream", tags=["stream"])


@router.post("")
async def chat(request: Request, body: ChatRequest) -> EventSourceResponse:
    async def event_generator():
        state = session_store.get(body.session_id)
        if state is None:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": f"Session '{body.session_id}' not found"}
                ),
            }
            return

        state.deps.clear_context()
        model = request.app.state.model

        try:
            async with agentic_rag.run_stream(
                body.message,
                model=model,
                message_history=state.messages,
                deps=state.deps,
            ) as result:
                async for chunk in result.stream_text(delta=True):
                    yield {"event": "delta", "data": json.dumps({"text": chunk})}

            state.messages.extend(result.all_messages())
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "contexts": state.deps.contexts or [],
                        "citations": state.deps.citations or [],
                    }
                ),
            }
        except Exception as e:
            logger.exception(f"Chat stream error: {e}")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return EventSourceResponse(event_generator())
