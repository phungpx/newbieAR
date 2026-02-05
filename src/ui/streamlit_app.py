"""Main Streamlit application for NewbieAR."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils import init_session_state, get_collections

# Configure page
st.set_page_config(
    page_title="NewbieAR - Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()

# Sidebar navigation with better organization
with st.sidebar:
    st.title("📚 NewbieAR")
    st.markdown("**Document RAG System**")
    st.markdown("---")

    # Navigation menu
    st.markdown("### Navigation")

    # Define pages with icons and descriptions
    pages = {
        "Home": {"icon": "🏠", "desc": "Overview and quick start"},
        "Chat": {"icon": "💬", "desc": "Query your documents"},
        "Ingest Documents": {"icon": "📄", "desc": "Upload and process documents"},
        "Collections": {"icon": "📊", "desc": "Manage vector store collections"},
        "Settings": {"icon": "⚙️", "desc": "Configure system settings"},
    }

    # Create navigation buttons
    selected_page = None
    for page_name, page_info in pages.items():
        if st.button(
            f"{page_info['icon']} {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.get("current_page") == page_name
                else "secondary"
            ),
        ):
            st.session_state.current_page = page_name
            st.rerun()

    # Use session state to track current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.markdown("---")

    # Status bar
    st.markdown("### Status")

    # Current collection
    selected_collection = st.session_state.get("selected_collection")
    if selected_collection:
        st.success(f"📦 **Collection:** {selected_collection}")
    else:
        st.warning("⚠️ No collection selected")

    # RAG mode
    rag_mode = st.session_state.get("rag_mode", "basic")
    st.info(f"🔧 **Mode:** {rag_mode.title()}")

    # Collection info
    collections = get_collections()
    if collections:
        st.metric("Collections", len(collections))

    st.markdown("---")

    # Quick actions
    st.markdown("### Quick Actions")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    if st.button("🗑️ Clear Session", use_container_width=True):
        # Clear chat history but keep config
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Session cleared!")
        st.rerun()

# Main content area
current_page = st.session_state.get("current_page", "Home")

# Route to pages
if current_page == "Home":
    _render_home_page()
elif current_page == "Chat":
    from src.ui.pages import chat

    chat.render()
elif current_page == "Ingest Documents":
    from src.ui.pages import ingest

    ingest.render()
elif current_page == "Collections":
    from src.ui.pages import collections

    collections.render()
elif current_page == "Settings":
    from src.ui.pages import settings

    settings.render()


def _render_home_page():
    """Render the home page."""
    st.title("Welcome to NewbieAR")
    st.markdown("### Document Retrieval-Augmented Generation System")

    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
        NewbieAR is a minimal, agentic RAG system for experimenting with retrieval, agents, 
        answer synthesis, and evaluation on your own document collections.
        """
        )

    with col2:
        st.markdown(
            """
        **Version:** 0.1.0  
        **Status:** 🟢 Active
        """
        )

    st.markdown("---")

    # Features grid
    st.markdown("### Features")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        #### 📄 Document Ingestion
        Upload and process PDFs and other documents with flexible chunking strategies.
        """
        )

    with col2:
        st.markdown(
            """
        #### 💬 Interactive Chat
        Query your documents using Basic RAG or Agentic RAG with citations and tool visibility.
        """
        )

    with col3:
        st.markdown(
            """
        #### 📊 Collection Management
        Manage your Qdrant vector store collections with ease.
        """
        )

    with col4:
        st.markdown(
            """
        #### ⚙️ Settings
        Configure RAG parameters and system settings to your needs.
        """
        )

    st.markdown("---")

    # Quick start guide
    st.markdown("### Quick Start Guide")

    with st.expander("📖 Step-by-step guide", expanded=True):
        st.markdown(
            """
        1. **Select or Create Collection**
           - Go to **Collections** page
           - Create a new collection or select an existing one
           - Set it as your active collection
        
        2. **Ingest Documents**
           - Go to **Ingest Documents** page
           - Upload your PDF, Markdown, or text files
           - Choose chunking strategy (Hybrid or Hierarchical)
           - Monitor progress and view ingestion history
        
        3. **Start Chatting**
           - Go to **Chat** page
           - Select RAG mode (Basic or Agentic)
           - Ask questions about your documents
           - View citations and tool calls
        
        4. **Configure Settings**
           - Go to **Settings** page
           - Adjust Top K, temperature, and other parameters
           - View system configuration
        """
        )

    st.markdown("---")

    # Current configuration
    st.markdown("### Current Configuration")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        collection_name = st.session_state.get("selected_collection", "Not set")
        st.metric("Collection", collection_name)

    with col2:
        rag_mode = st.session_state.get("rag_mode", "basic")
        st.metric("RAG Mode", rag_mode.title())

    with col3:
        top_k = st.session_state.get("top_k", 5)
        st.metric("Top K", top_k)

    with col4:
        collections = get_collections()
        st.metric("Total Collections", len(collections) if collections else 0)

    # System status
    st.markdown("---")
    st.markdown("### System Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        try:
            collections = get_collections()
            if collections:
                st.success(f"✅ Connected to Qdrant ({len(collections)} collections)")
            else:
                st.warning("⚠️ No collections found")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")

    with status_col2:
        from src.settings import settings

        if settings.llm_api_key:
            st.success("✅ LLM configured")
        else:
            st.warning("⚠️ LLM not configured")

    with status_col3:
        if settings.embedding_api_key:
            st.success("✅ Embeddings configured")
        else:
            st.warning("⚠️ Embeddings not configured")
