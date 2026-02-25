import sys
import time
from pathlib import Path

_project_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation")
st.caption("Evaluate golden test cases with deepeval metrics")

col1, col2 = st.columns(2)
with col1:
    goldens_dir = st.text_input("Goldens directory", value="data/goldens")
    collection_name = st.text_input("Collection name", value="research_papers")
with col2:
    retrieval_window_size = st.slider("Retrieval window size", min_value=1, max_value=20, value=5)
    threshold = st.slider("Metric threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    force_rerun = st.toggle("Force re-evaluation (ignore cache)")

start_btn = st.button("Start Evaluation")

if start_btn:
    with st.spinner("Submitting evaluation job…"):
        try:
            resp = client.post(
                api_url("/evaluation/jobs"),
                json={
                    "goldens_dir": goldens_dir,
                    "collection_name": collection_name,
                    "retrieval_window_size": retrieval_window_size,
                    "threshold": threshold,
                    "force_rerun": force_rerun,
                },
            )
            if resp.status_code != 202:
                st.error(f"Error {resp.status_code}: {resp.text}")
                st.stop()

            job_id = resp.json()["job_id"]
            st.info(f"Job submitted: `{job_id}`")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    # Poll until done or failed
    status_placeholder = st.empty()
    while True:
        try:
            poll = client.get(api_url(f"/evaluation/jobs/{job_id}"))
            if poll.status_code != 200:
                st.error(f"Poll error {poll.status_code}: {poll.text}")
                break
            body = poll.json()
            job_status = body["status"]
            status_placeholder.info(f"Status: **{job_status}**")

            if job_status == "done":
                result = body.get("result", {})
                evaluated = result.get("evaluated", 0)
                skipped = result.get("skipped", 0)
                avg_scores: dict = result.get("avg_scores", {})

                st.success(f"Evaluation complete — **{evaluated}** evaluated, **{skipped}** skipped")

                if avg_scores:
                    st.subheader("Average Scores")
                    cols = st.columns(len(avg_scores))
                    for col, (metric_name, score) in zip(cols, avg_scores.items()):
                        delta_color = "normal" if score >= threshold else "inverse"
                        col.metric(
                            label=metric_name,
                            value=f"{score:.3f}",
                            delta=f"{'≥' if score >= threshold else '<'} {threshold}",
                            delta_color=delta_color,
                        )
                break
            elif job_status == "failed":
                st.error(f"Evaluation failed: {body.get('error', 'Unknown error')}")
                break
        except Exception as exc:
            st.error(f"Poll request failed: {exc}")
            break

        time.sleep(3)
        st.rerun()
