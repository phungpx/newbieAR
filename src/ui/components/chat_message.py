"""UI components for rendering chat messages."""

import streamlit as st
from typing import Optional, Dict, Any
from src.models import CitedResponse, TurnMetadata
from src.ui.components.citations import render_citations_tab_view
from src.ui.components.tool_calls import render_tool_calls


def render_chat_message(message: Dict[str, Any]) -> None:
    """Render a chat message with all its components.

    Args:
        message: Message dictionary with role, content, and optional metadata
    """
    role = message.get("role", "user")
    content = message.get("content", "")

    with st.chat_message(role):
        # Display message content
        st.markdown(content)

        # Display cited response if available
        if "cited_response" in message and message["cited_response"]:
            cited_response = CitedResponse(**message["cited_response"])

            # Show citations and tool calls in tabs
            tab1, tab2, tab3 = st.tabs(["💬 Answer", "📚 Sources", "🔧 Tools"])

            with tab1:
                st.markdown(cited_response.answer)
                if cited_response.citations:
                    st.info(
                        f"📚 {len(cited_response.citations)} sources cited - see Sources tab"
                    )
                if cited_response.tool_calls:
                    st.info(
                        f"🔧 {len(cited_response.tool_calls)} tools used - see Tools tab"
                    )

            with tab2:
                if cited_response.citations:
                    from src.ui.components.citations import render_citations_section

                    render_citations_section(cited_response.citations)
                else:
                    st.info("No source documents were retrieved for this query")

            with tab3:
                if cited_response.tool_calls:
                    render_tool_calls(cited_response.tool_calls)
                else:
                    st.info("No tools were used for this query")

        # Legacy support: display retrieval info if available (old format)
        elif "retrieval_info" in message and message["retrieval_info"]:
            with st.expander("📄 Retrieved Documents"):
                from src.ui.utils import format_retrieval_info

                for i, info in enumerate(message["retrieval_info"]):
                    st.markdown(format_retrieval_info(info, i))

        # Display turn metadata if available
        if "turn_metadata" in message and message["turn_metadata"]:
            turn_metadata = TurnMetadata(**message["turn_metadata"])
            with st.expander("ℹ️ Turn Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Turn ID:** `{turn_metadata.turn_id}`")
                    st.markdown(f"**RAG Mode:** {turn_metadata.rag_mode}")
                with col2:
                    st.markdown(f"**Collection:** {turn_metadata.collection}")
                    st.markdown(
                        f"**Timestamp:** {turn_metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    )


def render_streaming_message(
    accumulated_text: str,
    tool_calls: Optional[list] = None,
    citations: Optional[list] = None,
) -> None:
    """Render a streaming message with optional tool calls and citations.

    Args:
        accumulated_text: Accumulated text so far
        tool_calls: Optional list of tool calls to display
        citations: Optional list of citations to display
    """
    st.markdown(accumulated_text + "▌")

    if tool_calls:
        st.markdown("**Tools used:**")
        for tool_call in tool_calls:
            st.markdown(f"- `{tool_call.tool_name}`")
