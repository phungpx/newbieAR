import time
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.dependencies import require_agents_permission
from src.api.models import (
    APIKey,
    AgentChatRequest,
    AgentChatResponse,
    Citation,
    ToolCallInfo,
    TokenUsage,
    TokenUsageBreakdown,
)
from src.api.services import session_manager
from pydantic_ai.messages import ModelResponse, ToolCallPart
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from src.retrieval.basic_rag import BasicRAG
from src.settings import settings

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/basic-rag/chat", response_model=AgentChatResponse)
async def basic_rag_chat(
    request: AgentChatRequest,
    api_key: APIKey = Depends(require_agents_permission),
):
    """
    Synchronous agentic chat with BasicRAG.
    Returns full response after completion.
    """
    start_time = time.time()

    try:
        # Initialize BasicRAG
        basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)

        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        else:
            # Try to get existing session; auto-create if not found
            session = session_manager.get_session(session_id, request.user_id)
            if not session:
                session_id = session_manager.create_session(request.user_id)

        # Get message history if requested
        messages = []
        if request.include_history:
            history = session_manager.get_history(session_id, request.user_id)
            # Convert to ModelMessage format (simplified)
            for msg in history:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Create dependencies
        deps = BasicRAGDependencies(
            basic_rag=basic_rag,
            top_k=request.top_k,
        )

        # Run agent
        result = await basic_rag_agent.run(
            request.message,
            message_history=messages,
            deps=deps,
        )

        # Extract response data (.output in pydantic-ai >= 1.x)
        response_text = result.output

        # Extract tool calls and citations from result
        tool_calls = []
        citations = []

        # Parse tool calls from ModelResponse messages
        for msg in result.all_messages():
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tool_calls.append(
                            ToolCallInfo(
                                tool=part.tool_name,
                                query=request.message,
                                results_count=request.top_k,
                                token_usage={"embedding_tokens": 8},
                                execution_time_ms=100,
                            )
                        )

        # Store conversation in session
        session_manager.add_message(session_id, role="user", content=request.message)
        session_manager.add_message(
            session_id,
            role="assistant",
            content=response_text,
            metadata={
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "citations": [c.model_dump() for c in citations],
            },
        )

        # Calculate token usage (simplified)
        total_tokens = len(request.message.split()) * 2 + len(response_text.split()) * 2

        elapsed_ms = int((time.time() - start_time) * 1000)

        return AgentChatResponse(
            message=response_text,
            tool_calls=tool_calls,
            citations=citations,
            session_id=session_id,
            user_id=request.user_id,
            token_usage=TokenUsage(
                total_prompt_tokens=len(request.message.split()) * 2,
                total_completion_tokens=len(response_text.split()) * 2,
                total_tokens=total_tokens,
                breakdown=TokenUsageBreakdown(
                    retrieval_embedding_tokens=8,
                    llm_prompt_tokens=len(request.message.split()) * 2,
                    llm_completion_tokens=len(response_text.split()) * 2,
                ),
            ),
            response_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"BasicRAG agent chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent chat failed: {str(e)}",
        )


@router.post("/basic-rag/stream")
async def basic_rag_stream(
    request: AgentChatRequest,
    api_key: APIKey = Depends(require_agents_permission),
):
    """
    Streaming agentic chat with BasicRAG.
    Returns Server-Sent Events stream.
    """

    async def event_generator():
        try:
            # Initialize BasicRAG
            basic_rag = BasicRAG(qdrant_collection_name=request.collection_name)

            # Get or create session
            session_id = request.session_id
            if not session_id:
                session_id = session_manager.create_session(request.user_id)
            else:
                # Try to get existing session; auto-create if not found
                session = session_manager.get_session(session_id, request.user_id)
                if not session:
                    session_id = session_manager.create_session(request.user_id)

            # Get message history
            messages = []
            if request.include_history:
                history = session_manager.get_history(session_id, request.user_id)
                for msg in history:
                    messages.append(
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        }
                    )

            # Create dependencies
            deps = BasicRAGDependencies(
                basic_rag=basic_rag,
                top_k=request.top_k,
            )

            start_time = time.time()
            accumulated_text = ""

            # Stream agent response
            async with basic_rag_agent.run_stream(
                request.message,
                message_history=messages,
                deps=deps,
            ) as result:
                # Stream tokens
                async for text_chunk in result.stream_text(delta=True):
                    accumulated_text += text_chunk
                    yield f"event: token\ndata: {json.dumps({'delta': text_chunk})}\n\n"

                # Send completion event
                elapsed_ms = int((time.time() - start_time) * 1000)
                total_tokens = (
                    len(request.message.split()) * 2 + len(accumulated_text.split()) * 2
                )

                done_data = {
                    "token_usage": {
                        "total_tokens": total_tokens,
                        "breakdown": {
                            "retrieval_embedding_tokens": 8,
                            "llm_prompt_tokens": len(request.message.split()) * 2,
                            "llm_completion_tokens": len(accumulated_text.split()) * 2,
                        },
                    },
                    "response_time_ms": elapsed_ms,
                    "session_id": session_id,
                }

                yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

                # Store in session
                session_manager.add_message(
                    session_id, role="user", content=request.message
                )
                session_manager.add_message(
                    session_id, role="assistant", content=accumulated_text
                )

        except Exception as e:
            logger.exception(f"Streaming failed: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Get conversation history for a session"""
    session = session_manager.get_session(session_id, user_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied",
        )

    messages = session_manager.get_history(
        session_id, user_id, limit=limit, offset=offset
    )

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "messages": messages,
        "total_messages": len(session.messages),
        "created_at": session.created_at.isoformat() + "Z",
    }


@router.post("/sessions")
async def create_session(
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Create new session explicitly"""
    session_id = session_manager.create_session(user_id)
    session = session_manager.get_session(session_id, user_id)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat() + "Z",
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """Delete session and clear history"""
    success = session_manager.delete_session(session_id, user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied",
        )

    return {"message": "Session deleted successfully"}


@router.get("/sessions")
async def list_sessions(
    user_id: str,
    api_key: APIKey = Depends(require_agents_permission),
):
    """List all sessions for a user"""
    sessions = session_manager.list_sessions(user_id)

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat() + "Z",
                "last_accessed_at": s.last_accessed_at.isoformat() + "Z",
                "message_count": len(s.messages),
            }
            for s in sessions
        ]
    }
