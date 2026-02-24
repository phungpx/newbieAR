import json
from typing import AsyncIterator, Optional
from src.api.agents.session_store import InMemorySessionStore
from src.agents.agentic_basic_rag import basic_rag_agent, BasicRAGDependencies, get_openai_model
from src.agents.agentic_graph_rag import graphiti_agent, GraphitiDependencies
from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval


class AgentService:
    def __init__(self, session_store: InMemorySessionStore):
        self.session_store = session_store

    async def stream_basic(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        session_id: Optional[str],
    ) -> AsyncIterator[str]:
        session_id, messages = self.session_store.get_or_create(session_id)
        model = get_openai_model()
        basic_rag = BasicRAG(qdrant_collection_name=collection_name)
        deps = BasicRAGDependencies(basic_rag=basic_rag, top_k=top_k)

        try:
            async with await basic_rag_agent.run_stream(
                query, model=model, message_history=messages, deps=deps
            ) as result:
                async for token in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            self.session_store.save(session_id, result.all_messages())
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'citations': []})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def stream_graph(
        self,
        query: str,
        top_k: int,
        session_id: Optional[str],
    ) -> AsyncIterator[str]:
        session_id, messages = self.session_store.get_or_create(session_id)
        model = get_openai_model()
        graph_retrieval = GraphRetrieval()
        deps = GraphitiDependencies(graph_retrieval=graph_retrieval, top_k=top_k)

        try:
            async with await graphiti_agent.run_stream(
                query, model=model, message_history=messages, deps=deps
            ) as result:
                async for token in result.stream_text(delta=True):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            self.session_store.save(session_id, result.all_messages())
            citations = deps.citations or []
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'citations': citations})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            await graph_retrieval.close()
