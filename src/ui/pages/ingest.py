"""Document ingestion page."""

import streamlit as st
import os
from pathlib import Path
from src.ui.utils import (
    get_newbiear_instance,
    save_uploaded_file,
    get_collections,
)
from src.models import ChunkStrategy
from src.settings import settings


def render():
    """Render the document ingestion page."""
    st.title("📄 Ingest Documents")
    st.markdown("Upload and process documents to add them to your vector store.")
    
    # Collection selector
    collections = get_collections()
    if not collections:
        st.warning("No collections found. Please create a collection first in the Collections page.")
        return
    
    selected_collection = st.selectbox(
        "Select Collection",
        collections,
        index=0 if st.session_state.get("selected_collection") in collections else 0,
    )
    
    if selected_collection != st.session_state.get("selected_collection"):
        st.session_state.selected_collection = selected_collection
    
    # Chunk strategy selector
    chunk_strategy = st.selectbox(
        "Chunk Strategy",
        [ChunkStrategy.HYBRID.value, ChunkStrategy.HIERARCHICAL.value],
        index=0,
        help="Hybrid: Combines multiple chunking approaches. Hierarchical: Uses hierarchical structure.",
    )
    
    # Directories configuration
    st.markdown("### Directories")
    col1, col2 = st.columns(2)
    with col1:
        documents_dir = st.text_input(
            "Documents Directory",
            value=st.session_state.get("documents_dir", "data/research_papers/files"),
            help="Directory where converted documents will be saved",
        )
    with col2:
        chunks_dir = st.text_input(
            "Chunks Directory",
            value=st.session_state.get("chunks_dir", "data/research_papers/chunks"),
            help="Directory where document chunks will be saved",
        )
    
    st.session_state.documents_dir = documents_dir
    st.session_state.chunks_dir = chunks_dir
    
    # File uploader
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "md", "txt", "docx"],
        help="Supported formats: PDF, Markdown, Text, DOCX",
    )
    
    if uploaded_file is not None:
        st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file to temp location
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    # Initialize newbieAR instance with selected chunk strategy
                    from src.main import newbieAR
                    app = newbieAR(
                        documents_dir=documents_dir,
                        chunks_dir=chunks_dir,
                        chunk_strategy=chunk_strategy,
                        qdrant_collection_name=selected_collection,
                    )
                    
                    # Ingest the file
                    result = app.ingest_file(temp_file_path)
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                    st.success("✅ Document ingested successfully!")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ Error ingesting document: {str(e)}")
                    # Clean up temp file on error
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
    
    # Batch upload section
    st.markdown("---")
    st.markdown("### Batch Upload")
    uploaded_files = st.file_uploader(
        "Choose multiple files",
        type=["pdf", "md", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload multiple files at once",
    )
    
    if uploaded_files:
        st.info(f"**Files selected:** {len(uploaded_files)}")
        
        if st.button("Ingest All Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            temp_files = []
            results = []
            
            try:
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                    
                    # Save uploaded file
                    temp_file_path = save_uploaded_file(uploaded_file)
                    temp_files.append(temp_file_path)
                    
                    # Initialize newbieAR instance with selected chunk strategy
                    from src.main import newbieAR
                    app = newbieAR(
                        documents_dir=documents_dir,
                        chunks_dir=chunks_dir,
                        chunk_strategy=chunk_strategy,
                        qdrant_collection_name=selected_collection,
                    )
                    
                    # Ingest the file
                    result = app.ingest_file(temp_file_path)
                    results.append({"file": uploaded_file.name, "result": result})
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                
                status_text.empty()
                st.success(f"✅ Successfully ingested {len(uploaded_files)} files!")
                
                # Display results
                with st.expander("View Results"):
                    for res in results:
                        st.markdown(f"**{res['file']}**")
                        st.json(res['result'])
                        
            except Exception as e:
                st.error(f"❌ Error during batch ingestion: {str(e)}")
                # Clean up temp files on error
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
