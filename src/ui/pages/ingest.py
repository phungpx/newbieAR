"""Document ingestion page."""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
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

    # Initialize ingestion history if needed
    if "ingestion_history" not in st.session_state:
        st.session_state.ingestion_history = []

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Configuration")

        # Collection selector
        collections = get_collections()
        if not collections:
            st.warning(
                "No collections found. Please create a collection first in the Collections page."
            )
            return

        selected_collection = st.selectbox(
            "Select Collection",
            collections,
            index=(
                0 if st.session_state.get("selected_collection") in collections else 0
            ),
            help="Select the vector store collection to ingest into",
        )

        if selected_collection != st.session_state.get("selected_collection"):
            st.session_state.selected_collection = selected_collection

        st.divider()

        # Chunk strategy selector
        st.markdown("### Chunk Strategy")
        chunk_strategy = st.selectbox(
            "Strategy",
            [ChunkStrategy.HYBRID.value, ChunkStrategy.HIERARCHICAL.value],
            index=0,
            help="Hybrid: Combines multiple chunking approaches. Hierarchical: Uses hierarchical structure.",
        )

        st.divider()

        # Directories configuration (collapsible)
        with st.expander("📁 Directory Settings"):
            documents_dir = st.text_input(
                "Documents Directory",
                value=st.session_state.get(
                    "documents_dir", "data/research_papers/files"
                ),
                help="Directory where converted documents will be saved",
            )
            chunks_dir = st.text_input(
                "Chunks Directory",
                value=st.session_state.get("chunks_dir", "data/research_papers/chunks"),
                help="Directory where document chunks will be saved",
            )

            st.session_state.documents_dir = documents_dir
            st.session_state.chunks_dir = chunks_dir

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Progress", "📜 History"])

    with tab1:
        _render_upload_tab(selected_collection, chunk_strategy)

    with tab2:
        _render_progress_tab()

    with tab3:
        _render_history_tab()


def _render_upload_tab(collection: str, chunk_strategy: str):
    """Render the upload tab.

    Args:
        collection: Selected collection
        chunk_strategy: Chunk strategy to use
    """
    st.markdown("### Single File Upload")

    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=["pdf", "md", "txt", "docx"],
        help="Supported formats: PDF, Markdown, Text, DOCX",
        key="single_file_uploader",
    )

    if uploaded_file is not None:
        # File preview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("File Type", Path(uploaded_file.name).suffix.upper())

        # Show file info
        with st.expander("📄 File Details"):
            st.json(
                {
                    "name": uploaded_file.name,
                    "size_bytes": uploaded_file.size,
                    "size_kb": f"{uploaded_file.size / 1024:.2f}",
                    "type": uploaded_file.type,
                    "collection": collection,
                    "chunk_strategy": chunk_strategy,
                }
            )

        if st.button("🚀 Ingest Document", type="primary", use_container_width=True):
            _ingest_single_file(uploaded_file, collection, chunk_strategy)

    st.divider()
    st.markdown("### Batch Upload")

    uploaded_files = st.file_uploader(
        "Choose multiple files to upload",
        type=["pdf", "md", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload multiple files at once",
        key="batch_file_uploader",
    )

    if uploaded_files:
        st.info(f"**{len(uploaded_files)} files selected**")

        # Show file list
        with st.expander("📋 View Selected Files"):
            for i, file in enumerate(uploaded_files, 1):
                st.markdown(f"{i}. **{file.name}** ({file.size / 1024:.2f} KB)")

        if st.button("🚀 Ingest All Files", type="primary", use_container_width=True):
            _ingest_batch_files(uploaded_files, collection, chunk_strategy)


def _render_progress_tab():
    """Render the progress tracking tab."""
    st.markdown("### Ingestion Progress")

    if "current_ingestion" not in st.session_state:
        st.info("No active ingestion. Start uploading files to see progress.")
        return

    ingestion = st.session_state.current_ingestion

    # Overall progress
    if ingestion.get("total_files"):
        progress = ingestion.get("completed_files", 0) / ingestion["total_files"]
        st.progress(progress)
        st.metric(
            "Progress",
            f"{ingestion['completed_files']}/{ingestion['total_files']} files",
        )

    # Current file being processed
    if ingestion.get("current_file"):
        st.markdown(f"**Processing:** {ingestion['current_file']}")
        if ingestion.get("current_status"):
            st.info(ingestion["current_status"])


def _render_history_tab():
    """Render the ingestion history tab."""
    st.markdown("### Ingestion History")

    if not st.session_state.ingestion_history:
        st.info("No ingestion history yet. Start uploading files to see history.")
        return

    # Show recent ingestions
    for i, entry in enumerate(reversed(st.session_state.ingestion_history[-10:]), 1):
        with st.expander(f"📄 {entry['file_name']} - {entry['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Status:** {entry['status']}")
            with col2:
                st.markdown(f"**Collection:** {entry['collection']}")
            with col3:
                st.markdown(f"**Strategy:** {entry['chunk_strategy']}")

            if entry.get("result"):
                st.json(entry["result"])

            if entry.get("error"):
                st.error(f"Error: {entry['error']}")


def _ingest_single_file(uploaded_file, collection: str, chunk_strategy: str):
    """Ingest a single file.

    Args:
        uploaded_file: Uploaded file object
        collection: Collection name
        chunk_strategy: Chunk strategy
    """
    documents_dir = st.session_state.get("documents_dir", "data/research_papers/files")
    chunks_dir = st.session_state.get("chunks_dir", "data/research_papers/chunks")

    progress_container = st.container()
    status_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    temp_file_path = None

    try:
        status_text.info(f"📤 Uploading {uploaded_file.name}...")
        progress_bar.progress(0.1)

        # Save uploaded file
        temp_file_path = save_uploaded_file(uploaded_file)
        progress_bar.progress(0.3)

        status_text.info(f"🔧 Initializing ingestion pipeline...")

        # Initialize newbieAR instance
        from src.main import newbieAR

        app = newbieAR(
            documents_dir=documents_dir,
            chunks_dir=chunks_dir,
            chunk_strategy=chunk_strategy,
            qdrant_collection_name=collection,
        )
        progress_bar.progress(0.5)

        status_text.info(f"⚙️ Processing document...")

        # Ingest the file
        result = app.ingest_file(temp_file_path)
        progress_bar.progress(0.9)

        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        progress_bar.progress(1.0)
        status_text.success(f"✅ {uploaded_file.name} ingested successfully!")

        # Add to history
        st.session_state.ingestion_history.append(
            {
                "file_name": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "collection": collection,
                "chunk_strategy": chunk_strategy,
                "result": result,
            }
        )

        # Show result
        with st.expander("📊 View Ingestion Result"):
            st.json(result)

        st.rerun()

    except Exception as e:
        error_msg = str(e)
        status_text.error(f"❌ Error ingesting {uploaded_file.name}: {error_msg}")

        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        # Add to history with error
        st.session_state.ingestion_history.append(
            {
                "file_name": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error",
                "collection": collection,
                "chunk_strategy": chunk_strategy,
                "error": error_msg,
            }
        )


def _ingest_batch_files(uploaded_files, collection: str, chunk_strategy: str):
    """Ingest multiple files.

    Args:
        uploaded_files: List of uploaded file objects
        collection: Collection name
        chunk_strategy: Chunk strategy
    """
    documents_dir = st.session_state.get("documents_dir", "data/research_papers/files")
    chunks_dir = st.session_state.get("chunks_dir", "data/research_papers/chunks")

    # Initialize progress tracking
    st.session_state.current_ingestion = {
        "total_files": len(uploaded_files),
        "completed_files": 0,
        "current_file": None,
        "current_status": None,
    }

    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()

    temp_files = []
    results = []
    errors = []

    try:
        for i, uploaded_file in enumerate(uploaded_files):
            st.session_state.current_ingestion["current_file"] = uploaded_file.name
            st.session_state.current_ingestion["current_status"] = (
                f"Processing {i+1}/{len(uploaded_files)}"
            )

            status_text.info(
                f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})..."
            )

            try:
                # Save uploaded file
                temp_file_path = save_uploaded_file(uploaded_file)
                temp_files.append(temp_file_path)

                # Initialize newbieAR instance
                from src.main import newbieAR

                app = newbieAR(
                    documents_dir=documents_dir,
                    chunks_dir=chunks_dir,
                    chunk_strategy=chunk_strategy,
                    qdrant_collection_name=collection,
                )

                # Ingest the file
                result = app.ingest_file(temp_file_path)
                results.append(
                    {
                        "file": uploaded_file.name,
                        "status": "success",
                        "result": result,
                    }
                )

                # Add to history
                st.session_state.ingestion_history.append(
                    {
                        "file_name": uploaded_file.name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "success",
                        "collection": collection,
                        "chunk_strategy": chunk_strategy,
                        "result": result,
                    }
                )

            except Exception as e:
                error_msg = str(e)
                errors.append(
                    {
                        "file": uploaded_file.name,
                        "error": error_msg,
                    }
                )

                # Add to history with error
                st.session_state.ingestion_history.append(
                    {
                        "file_name": uploaded_file.name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "error",
                        "collection": collection,
                        "chunk_strategy": chunk_strategy,
                        "error": error_msg,
                    }
                )

            # Update progress
            st.session_state.current_ingestion["completed_files"] = i + 1
            overall_progress.progress((i + 1) / len(uploaded_files))

        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Clear current ingestion
        st.session_state.current_ingestion = None

        # Show results
        with results_container:
            status_text.empty()

            if results:
                st.success(f"✅ Successfully ingested {len(results)} files!")

                with st.expander("📊 View Results"):
                    for res in results:
                        st.markdown(f"**{res['file']}**")
                        st.json(res.get("result", {}))

            if errors:
                st.warning(f"⚠️ {len(errors)} files failed to ingest")

                with st.expander("❌ View Errors"):
                    for err in errors:
                        st.error(f"**{err['file']}**: {err['error']}")

        st.rerun()

    except Exception as e:
        # Clean up temp files on error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        st.error(f"❌ Error during batch ingestion: {str(e)}")
        st.session_state.current_ingestion = None
