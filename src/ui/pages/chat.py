"""Chat interface page."""

import streamlit as st
import asyncio
from src.ui.utils import (
    get_basic_rag_instance,
    get_agentic_rag_deps,
    format_retrieval_info,
    get_collections,
    CitationFormatter,
)
from src.ui.components import render_citations_tab_view
from src.models import CitedResponse
from src.agents.agentic_basic_rag import basic_rag_agent
from src.settings import settings


def render():
    """Render the chat interface page."""
    st.title("💬 Chat with Documents")
    st.markdown("Query your documents using Basic RAG or Agentic RAG.")
    
    # Collection selector
    collections = get_collections()
    if not collections:
        st.warning("No collections found. Please create a collection first in the Collections page.")
        return
    
    # RAG mode selector
    col1, col2 = st.columns([1, 2])
    with col1:
        rag_mode = st.radio(
            "RAG Mode",
            ["basic", "agentic"],
            index=0 if st.session_state.get("rag_mode") == "basic" else 1,
            help="Basic RAG: Simple retrieval + generation. Agentic RAG: Uses agent with tools.",
        )
        st.session_state.rag_mode = rag_mode
    
    with col2:
        selected_collection = st.selectbox(
            "Collection",
            collections,
            index=0 if st.session_state.get("selected_collection") in collections else 0,
        )
        if selected_collection != st.session_state.get("selected_collection"):
            st.session_state.selected_collection = selected_collection
    
    # Top K selector
    top_k = st.slider(
        "Top K (Number of documents to retrieve)",
        min_value=1,
        max_value=20,
        value=st.session_state.get("top_k", 5),
        help="Number of most relevant documents to retrieve",
    )
    st.session_state.top_k = top_k
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if we have a cited response
            if "cited_response" in message and message["cited_response"]:
                cited_response = CitedResponse(**message["cited_response"])
                render_citations_tab_view(cited_response)
            else:
                # Legacy display for old messages
                st.markdown(message["content"])

                # Display retrieval info if available
                if "retrieval_info" in message and message["retrieval_info"]:
                    with st.expander("📄 Retrieved Documents"):
                        for i, info in enumerate(message["retrieval_info"]):
                            st.markdown(format_retrieval_info(info, i))
    
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
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        basic_rag = get_basic_rag_instance(selected_collection)
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
                            collection=selected_collection
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
            
            else:
                # Agentic RAG mode
                message_placeholder = st.empty()
                full_response = ""
                retrieval_infos = []

                try:
                    deps = get_agentic_rag_deps(selected_collection)

                    # Run agent with streaming
                    async def run_agent():
                        async with basic_rag_agent.run_stream(
                            prompt,
                            deps=deps,
                        ) as result:
                            accumulated_text = ""
                            async for message in result.stream_text(delta=True):
                                accumulated_text += message
                                message_placeholder.markdown(accumulated_text + "▌")
                            return accumulated_text, result

                    # Execute async function
                    answer, result = asyncio.run(run_agent())
                    message_placeholder.empty()  # Clear streaming placeholder

                    full_response = answer

                    # Extract retrieval info from tool calls
                    try:
                        # Get all messages from the result
                        all_messages = result.all_messages()

                        # Look for tool response messages
                        for msg in all_messages:
                            if hasattr(msg, 'kind') and msg.kind == 'response':
                                if hasattr(msg, 'content') and isinstance(msg.content, list):
                                    for part in msg.content:
                                        # Check if this is a tool return with our search results
                                        if hasattr(part, 'tool_name') and part.tool_name == 'search_basic_rag':
                                            if hasattr(part, 'content'):
                                                # The tool returns (retrieval_infos, answer)
                                                tool_result = part.content
                                                if isinstance(tool_result, tuple) and len(tool_result) == 2:
                                                    retrieval_infos = tool_result[0]
                                                    break
                    except Exception as e:
                        # If extraction fails, log but continue
                        import logging
                        logging.warning(f"Could not extract retrieval info: {e}")

                    # Create cited response
                    citation_formatter = CitationFormatter(retrieval_infos)
                    cited_response = citation_formatter.create_cited_response(
                        answer=full_response,
                        rag_mode="agentic",
                        collection=selected_collection
                    )

                    # Display with tab view
                    with message_placeholder.container():
                        render_citations_tab_view(cited_response)

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
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
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
