import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Callable, Awaitable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings

from src.settings import settings
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies
from src.agents.agentic_graph_rag import graphiti_agent, GraphitiDependencies

SUPPORTED_AGENTS: set[str] = {"basic-rag", "graph-rag"}

KNOWN_MODELS: list[str] = [
    "basic-rag",
    "graph-rag",
    "basic-rag/gemini-2.5-flash",
    "graph-rag/gemini-2.5-flash",
]


def parse_model_id(model_id: str) -> tuple[str, str | None]:
    """Split 'agent-type' or 'agent-type/llm-model' → (agent_type, llm_model).

    Raises ValueError for unknown agent types.
    """
    parts = model_id.split("/", 1)
    agent_type = parts[0]
    llm_model = parts[1] if len(parts) > 1 else None
    if agent_type not in SUPPORTED_AGENTS:
        raise ValueError(
            f"Unknown agent: {agent_type!r}. Must be one of {sorted(SUPPORTED_AGENTS)}"
        )
    return agent_type, llm_model


def make_llm_model(llm_model_name: str | None):
    """Return a pydantic-ai model instance for the given LLM name.

    - None or any non-gemini name → OpenAIChatModel using settings
    - "gemini-*" → GoogleModel via Vertex AI
    """
    model_settings = ModelSettings(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    if llm_model_name and llm_model_name.startswith("gemini-"):
        project = os.getenv("GOOGLE_VERTEX_PROJECT", "vns-durian-traceability")
        return GoogleModel(
            model_name=llm_model_name,
            provider=GoogleProvider(project=project, vertexai=True),
            settings=model_settings,
        )
    return OpenAIChatModel(
        model_name=llm_model_name or settings.llm_model,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        ),
        settings=model_settings,
    )


async def stream_response(
    agent_type: str,
    llm_model_name: str | None,
    user_input: str,
    prior_messages: list[ModelMessage],
    basic_rag,  # BasicRAG instance from app.state
    graph_retrieval,  # GraphRetrieval instance from app.state
    session_saver: Callable[[list[ModelMessage]], Awaitable[None]],
) -> AsyncGenerator[str, None]:
    """Yield SSE text chunks in OpenAI delta format, then save the session."""
    model = make_llm_model(llm_model_name)
    model_id = f"{agent_type}/{llm_model_name or settings.llm_model}"
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # First chunk announces the assistant role
    yield _chunk({"role": "assistant", "content": ""})

    if agent_type == "basic-rag":
        agent: Agent = basic_rag_agent
        deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=5)
    else:  # "graph-rag"
        agent = graphiti_agent
        deps = GraphitiDependencies(graph_retrieval=graph_retrieval, top_k=5)

    new_messages: list[ModelMessage] = []
    async with agent.run_stream(
        user_input,
        model=model,
        message_history=prior_messages,
        deps=deps,
    ) as result:
        async for delta in result.stream_text(delta=True):
            yield _chunk({"content": delta})
        new_messages = result.all_messages()

    # Persist the updated history before sending the terminal chunks
    await session_saver(prior_messages + new_messages)

    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"
