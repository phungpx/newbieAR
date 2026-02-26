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

# ── Database selector ──────────────────────────────────────
db_type = st.radio("Target database", ["Vector DB", "Graph DB"], horizontal=True)

# ── File uploader ──────────────────────────────────────────
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# ── File preview card ──────────────────────────────────────
if uploaded_file is not None:
    size_kb = len(uploaded_file.getvalue()) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
    st.info(f"📄 **{uploaded_file.name}** &nbsp;|&nbsp; {size_str} &nbsp;|&nbsp; PDF")

# ── Form options ───────────────────────────────────────────
if db_type == "Vector DB":
    collection_name = st.text_input("Collection name", value="research_papers")
    chunk_strategy = st.selectbox("Chunk strategy", ["hybrid", "hierarchical"], index=0)
else:
    chunk_strategy = st.selectbox("Chunk strategy", ["hierarchical", "hybrid"], index=0)

ingest_btn = st.button("Ingest", disabled=uploaded_file is None, type="primary")

# ── Ingestion + status ─────────────────────────────────────
if ingest_btn and uploaded_file is not None:
    with st.status(f"Ingesting **{uploaded_file.name}**…", expanded=True) as status:
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
                result = resp.json()
                st.write("✅ File uploaded")
                st.write("✅ PDF parsed")
                st.write("✅ Document chunked")
                st.write("✅ Chunks embedded")
                st.write("✅ Saved to store")
                status.update(
                    label=f"✅ {uploaded_file.name} ingested successfully",
                    state="complete",
                    expanded=True,
                )

                # ── Result summary card ────────────────────
                st.divider()
                st.subheader("Result Summary")
                if db_type == "Vector DB":
                    col1, col2 = st.columns(2)
                    col1.metric("Collection", result.get("collection_name", "—"))
                    col2.metric("Chunk Strategy", result.get("chunk_strategy", "—"))
                    st.caption(f"📁 Doc path: `{result.get('file_save_path', '—')}`")
                    st.caption(f"🧩 Chunk path: `{result.get('chunk_save_path', '—')}`")
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("File", result.get("filename", uploaded_file.name))
                    col2.metric("Chunk Strategy", result.get("chunk_strategy", "—"))
            else:
                status.update(
                    label=f"❌ Ingestion failed ({resp.status_code})",
                    state="error",
                    expanded=True,
                )
                st.error(f"Error {resp.status_code}: {resp.text}")

        except Exception as exc:
            status.update(label="❌ Request failed", state="error", expanded=True)
            st.error(f"Request failed: {exc}")

# ── Collection Summary (always visible, independent panel) ─
st.divider()
st.subheader("Collection Summary")
st.caption("Query live Qdrant stats for any collection")

with st.form("collection_summary_form"):
    query_collection = st.text_input(
        "Collection name",
        value="research_papers",
        key="summary_collection",
    )
    submitted = st.form_submit_button("View Summary")

if submitted:
    with st.spinner(f"Fetching stats for `{query_collection}`…"):
        try:
            resp = client.get(api_url(f"/ingest/collections/{query_collection}"))
            if resp.status_code == 200:
                info = resp.json()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vectors", info.get("vectors_count", 0))
                c2.metric("Dimensions", info.get("dimensions", "—"))
                c3.metric("Distance", info.get("distance", "—").capitalize())
                c4.metric("Status", info.get("status", "—").capitalize())
            elif resp.status_code == 404:
                st.warning(f"Collection `{query_collection}` not found.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
