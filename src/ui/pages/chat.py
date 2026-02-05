"""Chat interface page."""

import streamlit as st
import asyncio
from typing import Optional
from src.ui.utils import (
    get_basic_rag_instance,
    get_agentic_rag_deps,
    get_collections,
    CitationFormatter,
    run_agentic_rag_stream_with_metadata,
)
from src.ui.components.chat_message import render_chat_message
from src.ui.components.citations import render_citations_tab_view
from src.models import CitedResponse, TurnMetadata
from src.settings import settings


def render():
    """Render the chat interface page."""
    st.title("💬 Chat with Documents")
    st.markdown("Query your documents using Basic RAG or Agentic RAG.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Collection selector
        collections = get_collections()
        if not collections:
            st.warning("No collections found. Please create a collection first in the Collections page.")
            return
        
        selected_collection = st.selectbox(
            "Collection",
            collections,
            index=0 if st.session_state.get("selected_collection") in collections else 0,
            help="Select the vector store collection to query",
        )
        if selected_collection != st.session_state.get("selected_collection"):
            st.session_state.selected_collection = selected_collection
        
        # RAG mode selector
        rag_mode = st.radio(
            "RAG Mode",
            ["basic", "agentic"],
            index=0 if st.session_state.get("rag_mode") == "basic" else 1,
            help="Basic RAG: Simple retrieval + generation. Agentic RAG: Uses agent with tools.",
        )
        st.session_state.rag_mode = rag_mode
        
        # Top K selector
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5),
            help="Number of most relevant documents to retrieve",
        )
        st.session_state.top_k = top_k
        
        st.divider()
        
        # Display options
        st.markdown("### Display Options")
        show_tool_calls = st.checkbox(
            "Show Tool Calls",
            value=st.session_state.get("show_tool_calls", True),
            help="Display tool calls in agentic mode",
        )
        st.session_state.show_tool_calls = show_tool_calls
        
        show_citations = st.checkbox(
            "Show Citations",
            value=st.session_state.get("show_citations", True),
            help="Display source citations",
        )
        st.session_state.show_citations = show_citations
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if rag_mode == "basic":
                # Basic RAG mode
                _handle_basic_rag(prompt, selected_collection, top_k)
            else:
                # Agentic RAG mode
                _handle_agentic_rag(prompt, selected_collection, top_k)


def _handle_basic_rag(prompt: str, collection: str, top_k: int):
    """Handle Basic RAG query.
    
    Args:
        prompt: User query
        collection: Collection name
        top_k: Number of documents to retrieve
    """
    with st.spinner("Retrieving and generating answer..."):
        try:
            basic_rag = get_basic_rag_instance(collection)
            retrieval_infos, answer = basic_rag.generate(
                prompt,
                top_k=top_k,
                return_context=True,
            )

            # Create cited response
            citation_formatter = CitationFormatter(retrieval_infos)
            cited_response = citation_formatter.create_cited_response(
                answer=answer,
                rag_mode="basic",
                collection=collection
            )

            # Display with tab view
            render_citations_tab_view(cited_response)

            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieval_info": retrieval_infos,
                "cited_response": cited_response.model_dump(),
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })


def _handle_agentic_rag(prompt: str, collection: str, top_k: int):
    """Handle Agentic RAG query with streaming and metadata.
    
    Args:
        prompt: User query
        collection: Collection name
        top_k: Number of documents to retrieve
    """
    message_placeholder = st.empty()
    full_response = ""
    turn_metadata: Optional[TurnMetadata] = None
    tool_calls = []
    retrieval_infos = []

    try:
        # Run agent with streaming
        async def run_agent():
            nonlocal full_response, turn_metadata, tool_calls, retrieval_infos
            
            accumulated_text = ""
            async for text_delta, metadata in run_agentic_rag_stream_with_metadata(
                query=prompt,
                collection_name=collection,
                message_history=st.session_state.messages[:-1] if st.session_state.messages else None,
            ):
                if text_delta:
                    accumulated_text += text_delta
                    message_placeholder.markdown(accumulated_text + "▌")
                
                if metadata:
                    turn_metadata = metadata
                    tool_calls = metadata.tool_calls
                    # Extract retrieval info from tool calls
                    for tool_call in tool_calls:
                        if tool_call.tool_name == "search_basic_rag" and tool_call.result:
                            if isinstance(tool_call.result, tuple) and len(tool_call.result) >= 1:
                                retrieval_infos = tool_call.result[0]
                                break
            
            full_response = accumulated_text
            return full_response, turn_metadata

        # Execute async function
        answer, metadata = asyncio.run(run_agent())
        message_placeholder.empty()  # Clear streaming placeholder

        # Create cited response with metadata
        citation_formatter = CitationFormatter(retrieval_infos)
        cited_response = citation_formatter.create_cited_response(
            answer=full_response,
            rag_mode="agentic",
            collection=collection,
            tool_calls=tool_calls,
            turn_metadata=metadata,
        )

        # Display with tab view
        with message_placeholder.container():
            render_citations_tab_view(cited_response)

        # Add to history with full metadata
        message_data = {
            "role": "assistant",
            "content": full_response,
            "retrieval_info": retrieval_infos,
            "cited_response": cited_response.model_dump(),
        }
        
        if metadata:
            message_data["turn_metadata"] = metadata.model_dump()
        
        st.session_state.messages.append(message_data)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
        })
