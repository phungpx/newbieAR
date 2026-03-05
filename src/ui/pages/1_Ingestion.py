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
                chunk_count = result.get("chunk_count", 0)

                st.write("✅ File uploaded")
                st.write("✅ PDF parsed")
                st.write(f"✅ Document chunked ({chunk_count} chunks)")
                if db_type == "Vector DB":
                    st.write("✅ Chunks embedded")
                    st.write("✅ Saved to Qdrant")
                else:
                    st.write("✅ Episodes added to Neo4j")

                status.update(
                    label=f"✅ {uploaded_file.name} ingested successfully",
                    state="complete",
                    expanded=True,
                )

                # ── Result summary card ────────────────────
                st.divider()
                st.subheader("Result Summary")
                if db_type == "Vector DB":
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Collection", result.get("collection_name", "—"))
                    col2.metric("Chunk Strategy", result.get("chunk_strategy", "—"))
                    col3.metric("Chunks", chunk_count)
                    st.caption(f"📁 Doc path: `{result.get('file_save_path', '—')}`")
                    st.caption(f"🧩 Chunk path: `{result.get('chunk_save_path', '—')}`")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("File", result.get("filename", uploaded_file.name))
                    col2.metric("Chunk Strategy", result.get("chunk_strategy", "—"))
                    col3.metric("Chunks", chunk_count)

                # ── Chunk preview expander ─────────────────
                chunks = result.get("chunks", [])
                if chunks:
                    with st.expander(f"Show {len(chunks)} chunks"):
                        import pandas as pd

                        df = pd.DataFrame(
                            [
                                {
                                    "#": c["chunk_id"] + 1,
                                    "Tokens": c["text_tokens"],
                                    "Preview": c["text_preview"],
                                }
                                for c in chunks
                            ]
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)

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


# ── Bottom summary panels ──────────────────────────────────
st.divider()

if db_type == "Vector DB":
    # ── Collection Summary ─────────────────────────────────
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

                    # ── Delete Collection ──────────────────
                    st.divider()
                    if st.button(
                        f"🗑 Delete collection `{query_collection}`",
                        type="secondary",
                        key="delete_collection_btn",
                    ):
                        st.session_state["confirm_delete_collection"] = query_collection

                elif resp.status_code == 404:
                    st.warning(f"Collection `{query_collection}` not found.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    # Confirmation dialog for delete
    if st.session_state.get("confirm_delete_collection"):
        target = st.session_state["confirm_delete_collection"]
        st.warning(
            f"⚠️ Are you sure you want to permanently delete collection **{target}**? "
            "This cannot be undone."
        )
        col_yes, col_no = st.columns(2)
        if col_yes.button("Confirm Delete", type="primary", key="confirm_delete_yes"):
            try:
                resp = client.delete(api_url(f"/ingest/collections/{target}"))
                if resp.status_code == 200:
                    st.success(f"Collection `{target}` deleted.")
                else:
                    st.error(f"Delete failed: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            del st.session_state["confirm_delete_collection"]
            st.rerun()
        if col_no.button("Cancel", key="confirm_delete_no"):
            del st.session_state["confirm_delete_collection"]
            st.rerun()

else:
    # ── Graph Summary ──────────────────────────────────────
    st.subheader("Graph Summary")
    st.caption("Query live Neo4j stats")

    if st.button("View Summary", key="graph_summary_btn"):
        with st.spinner("Fetching Neo4j stats…"):
            try:
                resp = client.get(api_url("/ingest/graph/summary"))
                if resp.status_code == 200:
                    info = resp.json()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Nodes", info.get("nodes", 0))
                    c2.metric("Relationships", info.get("relationships", 0))
                    c3.metric("Communities", info.get("communities", 0))
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    # ── Clear Graph Data ───────────────────────────────────
    st.divider()
    if st.button("🗑 Clear Graph Data", type="secondary", key="clear_graph_btn"):
        st.session_state["confirm_clear_graph"] = True

    if st.session_state.get("confirm_clear_graph"):
        st.warning(
            "⚠️ Are you sure you want to clear **all Neo4j graph data**? "
            "This will delete all nodes and relationships and cannot be undone."
        )
        col_yes, col_no = st.columns(2)
        if col_yes.button("Confirm Clear", type="primary", key="confirm_clear_yes"):
            try:
                resp = client.post(api_url("/ingest/graph/clear"))
                if resp.status_code == 200:
                    st.success("All graph data cleared.")
                else:
                    st.error(f"Clear failed: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            del st.session_state["confirm_clear_graph"]
            st.rerun()
        if col_no.button("Cancel", key="confirm_clear_no"):
            del st.session_state["confirm_clear_graph"]
            st.rerun()
