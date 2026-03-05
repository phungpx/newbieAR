import sys
import math
import time
from pathlib import Path

_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "src").is_dir()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from src.ui.api_client import client, api_url

st.set_page_config(page_title="Synthesis", layout="wide")
st.title("Synthesis")
st.caption("Generate golden test cases from documents using deepeval Synthesizer")

# ── File uploader ──────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

# ── File preview cards ─────────────────────────────────────
if uploaded_files:
    st.markdown(f"**Uploaded Files ({len(uploaded_files)})**")
    cols_per_row = 3
    rows = math.ceil(len(uploaded_files) / cols_per_row)
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            file_idx = row * cols_per_row + col_idx
            if file_idx < len(uploaded_files):
                f = uploaded_files[file_idx]
                size_kb = len(f.getvalue()) / 1024
                size_str = (
                    f"{size_kb:.1f} KB"
                    if size_kb < 1024
                    else f"{size_kb / 1024:.2f} MB"
                )
                with cols[col_idx]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:8px; padding:12px; text-align:center;">
                            <div style="font-size:2em;">📄</div>
                            <div style="font-weight:bold; word-break:break-all; font-size:0.85em;">{f.name}</div>
                            <div style="color:gray; font-size:0.8em;">{size_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ── Synthesis options ──────────────────────────────────────
st.divider()
col1, col2 = st.columns(2)
with col1:
    output_dir = st.text_input("Output directory", value="data/goldens")
with col2:
    topic = st.selectbox("Topic", ["paper", "article"], index=0)
    num_contexts = st.slider("Number of contexts", min_value=1, max_value=50, value=5)
    context_size = st.slider("Context size", min_value=1, max_value=10, value=5)

start_btn = st.button(
    "Start Synthesis",
    disabled=not uploaded_files,
    type="primary",
)

# ── Two-step submit flow ───────────────────────────────────
if start_btn and uploaded_files:
    # Step 1: Upload files to server temp dir
    with st.spinner(f"Uploading {len(uploaded_files)} file(s)…"):
        try:
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_files
            ]
            upload_resp = client.post(
                api_url("/synthesis/upload"),
                files=files_payload,
            )
            if upload_resp.status_code != 200:
                st.error(f"Upload failed {upload_resp.status_code}: {upload_resp.text}")
                st.stop()
            upload_body = upload_resp.json()
            tmp_dir = upload_body["file_dir"]
            st.success(f"✅ {upload_body['file_count']} file(s) uploaded to server")
        except Exception as exc:
            st.error(f"Upload request failed: {exc}")
            st.stop()

    # Step 2: Submit synthesis job
    with st.spinner("Submitting synthesis job…"):
        try:
            resp = client.post(
                api_url("/synthesis/jobs"),
                json={
                    "file_dir": tmp_dir,
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

    # ── Polling loop ───────────────────────────────────────
    status_placeholder = st.empty()
    while True:
        try:
            poll = client.get(api_url(f"/synthesis/jobs/{job_id}"))
            if poll.status_code != 200:
                st.error(f"Poll error {poll.status_code}: {poll.text}")
                break
            body = poll.json()
            job_status = body["status"]

            if job_status == "pending":
                status_placeholder.info("⏳ Status: **pending** — waiting to start…")
            elif job_status == "running":
                status_placeholder.info("⚙️ Status: **running** — generating goldens…")

            if job_status == "done":
                status_placeholder.success("✅ Status: **done**")
                result = body.get("result", {})
                goldens_count = result.get("goldens_count", 0)
                files_count = len(uploaded_files)
                avg = goldens_count / files_count if files_count else 0

                st.divider()
                st.subheader("Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Goldens Generated", goldens_count)
                c2.metric("Files Processed", files_count)
                c3.metric("Avg Goldens / File", f"{avg:.1f}")
                st.caption(
                    f"📁 Output directory: `{result.get('output_dir', output_dir)}`"
                )
                break
            elif job_status == "failed":
                status_placeholder.error("❌ Status: **failed**")
                st.error(f"Synthesis failed: {body.get('error', 'Unknown error')}")
                break
        except Exception as exc:
            st.error(f"Poll request failed: {exc}")
            break

        time.sleep(3)
        st.rerun()
