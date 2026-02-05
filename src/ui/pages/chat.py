"""Chat interface page."""

import streamlit as st
import asyncio
from src.ui.utils import (
    get_basic_rag_instance,
    get_agentic_rag_deps,
    format_retrieval_info,
    get_collections,
)
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
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display retrieval info
                        with st.expander("📄 Retrieved Documents"):
                            for i, info in enumerate(retrieval_infos):
                                st.markdown(format_retrieval_info(info, i))
                        
                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "retrieval_info": retrieval_infos,
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
                    message_placeholder.markdown(answer)
                    full_response = answer
                    
                    # Try to extract retrieval info from tool calls
                    # The agent uses search_basic_rag tool which returns (retrieval_infos, answer)
                    # We can try to extract it from the result's tool calls
                    try:
                        # Check if we can get tool results from the result object
                        if hasattr(result, 'tool_calls') and result.tool_calls:
                            for tool_call in result.tool_calls:
                                if hasattr(tool_call, 'result') and tool_call.result:
                                    # The tool returns a tuple of (retrieval_infos, answer)
                                    if isinstance(tool_call.result, tuple) and len(tool_call.result) == 2:
                                        retrieval_infos = tool_call.result[0]
                    except Exception:
                        # If we can't extract, that's okay - we'll just show the answer
                        pass
                    
                    # Display retrieval info if available
                    if retrieval_infos:
                        with st.expander("📄 Retrieved Documents"):
                            for i, info in enumerate(retrieval_infos):
                                st.markdown(format_retrieval_info(info, i))
                    else:
                        with st.expander("📄 Retrieved Documents"):
                            st.info("Retrieval was performed by the agent. Detailed retrieval info may not be available.")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "retrieval_info": retrieval_infos,
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
