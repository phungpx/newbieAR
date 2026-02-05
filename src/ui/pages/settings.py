"""Settings page."""

import streamlit as st
from src.settings import settings


def render():
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure RAG parameters and view system settings.")
    
    # Editable RAG parameters
    st.markdown("### RAG Parameters")
    st.markdown("These settings affect how documents are retrieved and answers are generated.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5),
            help="Number of most relevant documents to retrieve",
        )
        st.session_state.top_k = top_k
    
    with col2:
        llm_temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.get("llm_temperature", settings.llm_temperature)),
            step=0.1,
            help="Controls randomness in LLM responses. Lower = more deterministic.",
        )
        st.session_state.llm_temperature = llm_temperature
    
    llm_max_tokens = st.number_input(
        "LLM Max Tokens",
        min_value=256,
        max_value=32768,
        value=st.session_state.get("llm_max_tokens", settings.llm_max_tokens),
        step=256,
        help="Maximum number of tokens in the LLM response",
    )
    st.session_state.llm_max_tokens = llm_max_tokens
    
    if st.button("Save Settings", type="primary"):
        st.success("✅ Settings saved to session state!")
    
    st.markdown("---")
    
    # Read-only configuration
    st.markdown("### System Configuration")
    st.markdown("Current system configuration (read-only).")
    
    # LLM Settings
    st.markdown("#### LLM Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("LLM Model", value=settings.llm_model, disabled=True)
    with col2:
        st.text_input("LLM Base URL", value=settings.llm_base_url or "Default", disabled=True)
    with col3:
        st.text_input("LLM API Key", value="***" if settings.llm_api_key else "Not set", disabled=True)
    
    # Embedding Settings
    st.markdown("#### Embedding Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.text_input("Embedding Model", value=settings.embedding_model, disabled=True)
    with col2:
        st.text_input("Embedding Base URL", value=settings.embedding_base_url, disabled=True)
    with col3:
        st.text_input("Embedding Dimensions", value=str(settings.embedding_dimensions), disabled=True)
    with col4:
        st.text_input("Embedding API Key", value="***" if settings.embedding_api_key else "Not set", disabled=True)
    
    # Qdrant Settings
    st.markdown("#### Qdrant Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Qdrant URI", value=settings.qdrant_uri, disabled=True)
    with col2:
        st.text_input("Qdrant API Key", value="***" if settings.qdrant_api_key else "Not set", disabled=True)
    with col3:
        st.text_input("Default Collection", value=settings.qdrant_collection_name, disabled=True)
    
    # Session State Info
    st.markdown("---")
    st.markdown("### Session State")
    with st.expander("View Session State"):
        session_data = {
            "selected_collection": st.session_state.get("selected_collection"),
            "rag_mode": st.session_state.get("rag_mode"),
            "top_k": st.session_state.get("top_k"),
            "llm_temperature": st.session_state.get("llm_temperature"),
            "llm_max_tokens": st.session_state.get("llm_max_tokens"),
            "documents_dir": st.session_state.get("documents_dir"),
            "chunks_dir": st.session_state.get("chunks_dir"),
            "chat_history_count": len(st.session_state.get("chat_history", [])),
        }
        st.json(session_data)
    
    # Reset session state
    st.markdown("---")
    st.markdown("### Reset Session")
    st.warning("⚠️ This will clear all session state including chat history.")
    
    if st.button("Reset Session State", type="secondary"):
        # Clear chat-related state but keep some defaults
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.success("✅ Session state reset!")
        st.rerun()
