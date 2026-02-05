"""Collections management page."""

import streamlit as st
from src.ui.utils import get_collections, get_collection_info
from src.deps import QdrantVectorStore
from src.settings import settings


def render():
    """Render the collections management page."""
    st.title("📊 Collections")
    st.markdown("Manage your Qdrant vector store collections.")
    
    # Get all collections
    collections = get_collections()
    
    # Create new collection section
    st.markdown("### Create New Collection")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_collection_name = st.text_input(
            "Collection Name",
            placeholder="e.g., my-documents",
            help="Enter a unique name for the new collection",
        )
    
    with col2:
        embedding_size = st.number_input(
            "Embedding Size",
            min_value=128,
            max_value=4096,
            value=settings.embedding_dimensions,
            step=128,
            help="Dimension of the embedding vectors",
        )
    
    if st.button("Create Collection", type="primary"):
        if not new_collection_name:
            st.error("Please enter a collection name.")
        elif new_collection_name in collections:
            st.error(f"Collection '{new_collection_name}' already exists.")
        else:
            try:
                vector_store = QdrantVectorStore(
                    uri=settings.qdrant_uri,
                    api_key=settings.qdrant_api_key,
                )
                vector_store.create_collection(
                    collection_name=new_collection_name,
                    embedding_size=embedding_size,
                    distance="cosine",
                )
                st.success(f"✅ Collection '{new_collection_name}' created successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error creating collection: {str(e)}")
    
    st.markdown("---")
    
    # List existing collections
    st.markdown("### Existing Collections")
    
    if not collections:
        st.info("No collections found. Create a new collection above.")
        return
    
    # Collection selector
    selected_collection = st.selectbox(
        "Select Collection",
        collections,
        index=0 if st.session_state.get("selected_collection") in collections else 0,
        help="Select a collection to view details or set as active",
    )
    
    # Set as active collection
    if st.button("Set as Active Collection"):
        st.session_state.selected_collection = selected_collection
        st.success(f"✅ '{selected_collection}' is now the active collection.")
    
    # Display collection information
    st.markdown("### Collection Information")
    
    try:
        collection_info = get_collection_info(selected_collection)
        
        if collection_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Points Count", collection_info.get("points_count", 0))
            
            with col2:
                st.metric("Vectors Count", collection_info.get("vectors_count", 0))
            
            with col3:
                config = collection_info.get("config", {})
                if config:
                    vector_params = config.get("params", {}).get("vectors", {})
                    if isinstance(vector_params, dict):
                        size = vector_params.get("size", "N/A")
                    else:
                        size = getattr(vector_params, "size", "N/A")
                    st.metric("Vector Size", size)
            
            # Display full config
            with st.expander("View Full Configuration"):
                st.json(collection_info)
        else:
            st.warning("Could not retrieve collection information.")
            
    except Exception as e:
        st.error(f"Error retrieving collection info: {str(e)}")
    
    # Delete collection section
    st.markdown("---")
    st.markdown("### Delete Collection")
    st.warning("⚠️ This action cannot be undone!")
    
    delete_collection = st.selectbox(
        "Collection to Delete",
        collections,
        key="delete_collection_selector",
        help="Select a collection to delete",
    )
    
    if st.button("Delete Collection", type="secondary"):
        if delete_collection == selected_collection:
            st.error("Cannot delete the currently selected collection. Please select a different collection first.")
        else:
            try:
                vector_store = QdrantVectorStore(
                    uri=settings.qdrant_uri,
                    api_key=settings.qdrant_api_key,
                )
                vector_store.delete_collection(delete_collection)
                st.success(f"✅ Collection '{delete_collection}' deleted successfully!")
                
                # Update selected collection if it was deleted
                if st.session_state.get("selected_collection") == delete_collection:
                    remaining = get_collections()
                    if remaining:
                        st.session_state.selected_collection = remaining[0]
                    else:
                        st.session_state.selected_collection = None
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error deleting collection: {str(e)}")
