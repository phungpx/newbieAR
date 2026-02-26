import sys
from pathlib import Path

_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "src").is_dir()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Ingestion", layout="wide")
st.title("Ingestion")
st.caption("Upload a PDF into Vector DB (Qdrant) or Graph DB (Neo4j via Graphiti)")

db_type = st.radio("Target database", ["Vector DB", "Graph DB"], horizontal=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if db_type == "Vector DB":
    collection_name = st.text_input("Collection name", value="research_papers")
    chunk_strategy = st.selectbox("Chunk strategy", ["hybrid", "hierarchical"], index=0)
else:
    chunk_strategy = st.selectbox("Chunk strategy", ["hierarchical", "hybrid"], index=0)

ingest_btn = st.button("Ingest", disabled=uploaded_file is None)

if ingest_btn and uploaded_file is not None:
    with st.spinner("Ingesting…"):
        try:
            if db_type == "Vector DB":
                resp = client.post(
                    api_url("/ingest/vector"),
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={
                        "collection_name": collection_name,
                        "chunk_strategy": chunk_strategy,
                    },
                )
            else:
                resp = client.post(
                    api_url("/ingest/graph"),
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={"chunk_strategy": chunk_strategy},
                )

            if resp.status_code == 200:
                st.success("Ingestion complete")
                st.json(resp.json())
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
