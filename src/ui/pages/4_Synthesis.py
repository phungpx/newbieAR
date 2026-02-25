import time
import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Synthesis", layout="wide")
st.title("Synthesis")
st.caption("Generate golden test cases from documents using deepeval Synthesizer")

col1, col2 = st.columns(2)
with col1:
    file_dir = st.text_input("File directory", value="data/papers/files")
    output_dir = st.text_input("Output directory", value="data/goldens")
with col2:
    topic = st.selectbox("Topic", ["paper", "article"], index=0)
    num_contexts = st.slider("Number of contexts", min_value=1, max_value=50, value=5)
    context_size = st.slider("Context size", min_value=1, max_value=10, value=5)

start_btn = st.button("Start Synthesis")

if start_btn:
    with st.spinner("Submitting synthesis job…"):
        try:
            resp = client.post(
                api_url("/synthesis/jobs"),
                json={
                    "file_dir": file_dir,
                    "output_dir": output_dir,
                    "topic": topic,
                    "num_contexts": num_contexts,
                    "context_size": context_size,
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
            poll = client.get(api_url(f"/synthesis/jobs/{job_id}"))
            if poll.status_code != 200:
                st.error(f"Poll error {poll.status_code}: {poll.text}")
                break
            body = poll.json()
            job_status = body["status"]
            status_placeholder.info(f"Status: **{job_status}**")

            if job_status == "done":
                result = body.get("result", {})
                st.success(
                    f"Synthesis complete — **{result.get('goldens_count', 0)}** goldens "
                    f"saved to `{result.get('output_dir', output_dir)}`"
                )
                break
            elif job_status == "failed":
                st.error(f"Synthesis failed: {body.get('error', 'Unknown error')}")
                break
        except Exception as exc:
            st.error(f"Poll request failed: {exc}")
            break

        time.sleep(3)
        st.rerun()
