from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatCompletionRequest
from src.api.services.agent_runner import parse_model_id, stream_response
from src.api.services.session import derive_session_key, load_messages, save_messages

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    try:
        agent_type, llm_model = parse_model_id(body.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    redis = request.app.state.redis
    session_key = derive_session_key(body.model, body.messages)
    prior_messages = await load_messages(redis, session_key)
    user_input = body.messages[-1].content
    ttl: int = request.app.state.session_ttl

    async def saver(messages):
        await save_messages(redis, session_key, messages, ttl=ttl)

    return StreamingResponse(
        stream_response(
            agent_type=agent_type,
            llm_model_name=llm_model,
            user_input=user_input,
            prior_messages=prior_messages,
            basic_rag=request.app.state.basic_rag,
            graph_retrieval=request.app.state.graph_retrieval,
            session_saver=saver,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
