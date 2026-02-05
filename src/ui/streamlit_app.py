"""Main Streamlit application for NewbieAR."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils import init_session_state

# Configure page
st.set_page_config(
    page_title="NewbieAR - Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

# Sidebar navigation
st.sidebar.title("📚 NewbieAR")
st.sidebar.markdown("Document RAG System")

# Page selection
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Ingest Documents", "Chat", "Collections", "Settings"],
    index=0,
)

# Display current collection in sidebar
if st.session_state.get("selected_collection"):
    st.sidebar.info(f"**Collection:** {st.session_state.selected_collection}")

# Route to pages
if page == "Home":
    st.title("Welcome to NewbieAR")
    st.markdown("### Document Retrieval-Augmented Generation System")
    
    st.markdown("""
    NewbieAR is a minimal, agentic RAG system for experimenting with retrieval, agents, 
    answer synthesis, and evaluation on your own document collections.
    
    **Features:**
    - 📄 **Document Ingestion**: Upload and process PDFs and other documents
    - 💬 **Interactive Chat**: Query your documents using Basic RAG or Agentic RAG
    - 📊 **Collection Management**: Manage your Qdrant vector store collections
    - ⚙️ **Settings**: Configure RAG parameters and system settings
    
    **Quick Start:**
    1. Go to **Collections** to select or create a collection
    2. Go to **Ingest Documents** to upload your documents
    3. Go to **Chat** to start querying your documents
    4. Adjust settings in **Settings** as needed
    """)
    
    st.markdown("---")
    st.markdown("### Current Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection", st.session_state.get("selected_collection", "Not set"))
    with col2:
        st.metric("RAG Mode", st.session_state.get("rag_mode", "basic").title())
    with col3:
        st.metric("Top K", st.session_state.get("top_k", 5))

elif page == "Ingest Documents":
    from src.ui.pages import ingest
    ingest.render()

elif page == "Chat":
    from src.ui.pages import chat
    chat.render()

elif page == "Collections":
    from src.ui.pages import collections
    collections.render()

elif page == "Settings":
    from src.ui.pages import settings
    settings.render()
