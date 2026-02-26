import sys
from pathlib import Path

_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "src").is_dir()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Retrieval", layout="wide")
st.title("Retrieval")
st.caption("Query documents with BasicRAG (vector) or GraphRAG (graph)")

search_type = st.radio(
    "Search mode", ["Vector (BasicRAG)", "Graph (GraphRAG)"], horizontal=True
)

query = st.text_input("Query", placeholder="Ask a question about your documents…")
top_k = st.slider("Top K results", min_value=1, max_value=50, value=5)

if search_type == "Vector (BasicRAG)":
    collection_name = st.text_input("Collection name", value="research_papers")
    score_threshold = st.slider(
        "Score threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )
    rerank = st.toggle("Enable cross-encoder reranking")

search_btn = st.button("Search", disabled=not query.strip())

if search_btn and query.strip():
    with st.spinner("Searching…"):
        try:
            if search_type == "Vector (BasicRAG)":
                resp = client.post(
                    api_url("/retrieve/vector"),
                    json={
                        "query": query,
                        "collection_name": collection_name,
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "rerank": rerank,
                    },
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    st.success(f"{len(results)} result(s) found")
                    for i, r in enumerate(results):
                        with st.expander(
                            f"#{i + 1} — {r['source']} (score: {r['score']:.4f})"
                        ):
                            st.markdown(r["content"])
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            else:
                resp = client.post(
                    api_url("/retrieve/graph"),
                    json={"query": query, "top_k": top_k},
                )
                if resp.status_code == 200:
                    body = resp.json()
                    contexts = body.get("contexts", [])
                    citations = body.get("citations", [])
                    st.success(f"{len(contexts)} context(s) found")
                    for i, ctx in enumerate(contexts):
                        with st.expander(f"Context #{i + 1}"):
                            st.markdown(ctx)
                    if citations:
                        st.subheader("Citations")
                        for c in citations:
                            st.markdown(f"- `{c}`")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
