import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="newbieAR — Pipeline Explorer", layout="wide")

st.title("newbieAR — Pipeline Explorer")
st.subheader("End-to-end Agentic RAG developer tool")

# Sidebar health check
with st.sidebar:
    st.header("Backend")
    try:
        resp = client.get(api_url("/health").replace("/api/v1", ""))
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            st.success("Connected")
        else:
            st.error(f"Unexpected response: {resp.status_code}")
    except Exception as exc:
        st.error(f"Cannot reach backend: {exc}")

# Pipeline navigation table
st.markdown(
    """
## Pipeline Stages

| # | Page | Description |
|---|------|-------------|
| 1 | [Ingestion](Ingestion) | Upload PDFs into vector DB or graph DB |
| 2 | [Retrieval](Retrieval) | Query with BasicRAG or GraphRAG |
| 3 | [Agent](Agent) | Chat with the agentic RAG (streaming) |
| 4 | [Synthesis](Synthesis) | Generate golden test cases from documents |
| 5 | [Evaluation](Evaluation) | Evaluate goldens with deepeval metrics |
"""
)
