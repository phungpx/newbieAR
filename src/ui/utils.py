"""Utility functions for Streamlit UI."""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List
import streamlit as st

from src.main import newbieAR
from src.retrieval.basic_rag import BasicRAG
from src.agents.agentic_basic_rag import (
    basic_rag_agent,
    BasicRAGDependencies,
    get_openai_model,
    run_agent_with_metadata,
    run_agent_stream_with_metadata,
)
from src.models import (
    RetrievalInfo,
    CitationInfo,
    CitedResponse,
    TurnMetadata,
    ToolCallInfo,
)
from src.settings import settings
from src.deps import QdrantVectorStore


class CitationFormatter:
    """Formats retrieval results into citations."""

    def __init__(self, retrieval_infos: List[RetrievalInfo]):
        """Initialize with retrieval information.

        Args:
            retrieval_infos: List of retrieved documents with scores
        """
        # Handle None or empty input
        self.retrieval_infos = retrieval_infos if retrieval_infos else []

    def create_citations(self) -> List[CitationInfo]:
        """Create citation list sorted by relevance score.

        Returns:
            List of CitationInfo objects with sequential citation numbers
        """
        if not self.retrieval_infos:
            return []

        # Sort by score (highest first), handle missing scores
        sorted_infos = sorted(
            self.retrieval_infos, key=lambda x: getattr(x, "score", 0.0), reverse=True
        )

        # Create citations with sequential numbers
        citations = []
        for idx, info in enumerate(sorted_infos, start=1):
            try:
                citation = CitationInfo(
                    citation_number=idx,
                    content=info.content if info.content else "No content available",
                    source=self._sanitize_source(getattr(info, "source", "")),
                    score=getattr(info, "score", 0.0),
                )
                citations.append(citation)
            except Exception as e:
                # Log error but continue processing other citations
                import logging

                logging.warning(f"Failed to create citation {idx}: {e}")
                continue

        return citations

    def create_cited_response(
        self,
        answer: str,
        rag_mode: str = "basic",
        collection: str = "",
        tool_calls: Optional[List[ToolCallInfo]] = None,
        turn_metadata: Optional[TurnMetadata] = None,
    ) -> CitedResponse:
        """Create a complete cited response.

        Args:
            answer: The generated answer text
            rag_mode: RAG mode used (basic/agentic/graph)
            collection: Collection name used
            tool_calls: Optional list of tool calls
            turn_metadata: Optional turn metadata

        Returns:
            CitedResponse with answer, citations, and tool calls
        """
        citations = self.create_citations()

        return CitedResponse(
            answer=answer if answer else "No answer generated",
            citations=citations,
            tool_calls=tool_calls or [],
            turn_metadata=turn_metadata,
            metadata={
                "rag_mode": rag_mode,
                "collection": collection,
                "citation_count": len(citations),
                "tool_call_count": len(tool_calls) if tool_calls else 0,
            },
        )

    @staticmethod
    def _sanitize_source(source: str) -> str:
        """Sanitize source filename for display.

        Args:
            source: Raw source path/filename

        Returns:
            Cleaned source name
        """
        if not source or not isinstance(source, str):
            return "Unknown source"

        # Get basename only
        try:
            basename = Path(source).name
        except Exception:
            # Fallback if path is malformed
            basename = str(source)

        # Truncate if too long
        max_length = 50
        if len(basename) > max_length:
            basename = basename[: max_length - 3] + "..."

        return basename


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = settings.qdrant_collection_name

    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "basic"

    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = settings.llm_temperature

    if "llm_max_tokens" not in st.session_state:
        st.session_state.llm_max_tokens = settings.llm_max_tokens

    if "documents_dir" not in st.session_state:
        st.session_state.documents_dir = "data/research_papers/files"

    if "chunks_dir" not in st.session_state:
        st.session_state.chunks_dir = "data/research_papers/chunks"


def get_newbiear_instance(
    collection_name: Optional[str] = None,
    documents_dir: Optional[str] = None,
    chunks_dir: Optional[str] = None,
) -> newbieAR:
    """Initialize and return a newbieAR instance."""
    collection = (
        collection_name
        or st.session_state.get("selected_collection")
        or settings.qdrant_collection_name
    )
    docs_dir = documents_dir or st.session_state.get(
        "documents_dir", "data/research_papers/files"
    )
    chunks_dir_path = chunks_dir or st.session_state.get(
        "chunks_dir", "data/research_papers/chunks"
    )

    return newbieAR(
        documents_dir=docs_dir,
        chunks_dir=chunks_dir_path,
        qdrant_collection_name=collection,
    )


def get_basic_rag_instance(collection_name: Optional[str] = None) -> BasicRAG:
    """Initialize and return a BasicRAG instance."""
    collection = (
        collection_name
        or st.session_state.get("selected_collection")
        or settings.qdrant_collection_name
    )
    return BasicRAG(qdrant_collection_name=collection)


def get_agentic_rag_deps(collection_name: Optional[str] = None) -> BasicRAGDependencies:
    """Get dependencies for Agentic RAG."""
    basic_rag = get_basic_rag_instance(collection_name)
    top_k = st.session_state.get("top_k", 5)
    return BasicRAGDependencies(basic_rag=basic_rag, top_k=top_k)


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location and return path."""
    # Create temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def format_retrieval_info(retrieval_info: RetrievalInfo, index: int) -> str:
    """Format a RetrievalInfo object for display."""
    return f"""
**Rank {index + 1}** | Score: {retrieval_info.score:.4f} | Source: {retrieval_info.source}

{retrieval_info.content[:500]}{'...' if len(retrieval_info.content) > 500 else ''}
"""


def format_chat_message(role: str, content: str) -> str:
    """Format a chat message for display."""
    return f"**{role.capitalize()}:**\n\n{content}"


def get_collections() -> List[str]:
    """Get list of all collections from Qdrant."""
    try:
        vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        collections_response = vector_store.client.get_collections()
        return [col.name for col in collections_response.collections]
    except Exception as e:
        # Don't use st.error here as this might be called outside streamlit context
        # Return empty list and let the calling code handle the error
        return []


def get_collection_info(collection_name: str) -> dict:
    """Get information about a collection."""
    try:
        vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        collection_info = vector_store.client.get_collection(collection_name)
        vectors_count = collection_info.points_count
        if hasattr(collection_info, "vectors_count"):
            vectors_count = collection_info.vectors_count
        elif hasattr(collection_info, "config") and hasattr(
            collection_info.config, "params"
        ):
            # Try to get from config
            pass

        return {
            "name": collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": vectors_count,
            "config": (
                collection_info.config.model_dump()
                if hasattr(collection_info.config, "model_dump")
                else str(collection_info.config)
            ),
        }
    except Exception as e:
        # Don't use st.error here as this might be called outside streamlit context
        return {}


async def run_agentic_rag_streaming(
    query: str,
    collection_name: Optional[str] = None,
) -> Tuple[List[RetrievalInfo], str]:
    """Run agentic RAG with streaming and return results."""
    deps = get_agentic_rag_deps(collection_name)

    # Run the agent
    result = await basic_rag_agent.run(query, deps=deps)

    # Extract retrieval info from tool calls if available
    retrieval_infos = []
    answer = ""

    # Get messages from result
    messages = result.all_messages()

    # Try to extract answer from messages
    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                answer += msg.content
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, str):
                        answer += part

    # If we have tool results, extract retrieval info
    # This is a simplified version - in practice, you'd parse tool results
    return retrieval_infos, answer or "No answer generated."


def run_agentic_rag_sync(
    query: str,
    collection_name: Optional[str] = None,
) -> Tuple[List[RetrievalInfo], str]:
    """Run agentic RAG synchronously (wrapper for async function)."""
    return asyncio.run(run_agentic_rag_streaming(query, collection_name))


async def run_agentic_rag_with_metadata(
    query: str,
    collection_name: Optional[str] = None,
    message_history: Optional[list] = None,
) -> Tuple[str, TurnMetadata]:
    """Run agentic RAG and return answer with full metadata.

    Args:
        query: User query
        collection_name: Collection name
        message_history: Optional message history

    Returns:
        Tuple of (answer_text, turn_metadata)
    """
    deps = get_agentic_rag_deps(collection_name)
    collection = collection_name or st.session_state.get("selected_collection", "")

    return await run_agent_with_metadata(
        query=query,
        deps=deps,
        message_history=message_history,
        collection=collection,
    )


async def run_agentic_rag_stream_with_metadata(
    query: str,
    collection_name: Optional[str] = None,
    message_history: Optional[list] = None,
):
    """Run agentic RAG with streaming and yield text deltas with final metadata.

    Args:
        query: User query
        collection_name: Collection name
        message_history: Optional message history

    Yields:
        Tuples of (text_delta, metadata_or_none)
    """
    deps = get_agentic_rag_deps(collection_name)
    collection = collection_name or st.session_state.get("selected_collection", "")

    async for text_delta, metadata in run_agent_stream_with_metadata(
        query=query,
        deps=deps,
        message_history=message_history,
        collection=collection,
    ):
        yield text_delta, metadata


def format_tool_call(tool_call: ToolCallInfo) -> str:
    """Format a tool call for display.

    Args:
        tool_call: Tool call to format

    Returns:
        Formatted string
    """
    status_icon = (
        "✅"
        if tool_call.status == "success"
        else "❌" if tool_call.status == "error" else "⏳"
    )
    time_str = f" ({tool_call.execution_time:.2f}s)" if tool_call.execution_time else ""
    return f"{status_icon} `{tool_call.tool_name}`{time_str}"


def format_turn_metadata(turn_metadata: TurnMetadata) -> str:
    """Format turn metadata for display.

    Args:
        turn_metadata: Turn metadata to format

    Returns:
        Formatted string
    """
    return f"Turn {turn_metadata.turn_id[:8]}... | {turn_metadata.rag_mode} | {len(turn_metadata.tool_calls)} tools | {len(turn_metadata.citations)} citations"
