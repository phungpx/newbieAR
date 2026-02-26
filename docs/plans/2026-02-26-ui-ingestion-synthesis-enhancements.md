# UI Ingestion & Synthesis Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance Ingestion page with file preview, step-by-step status, result card, and live collection summary; enhance Synthesis page with multi-file upload, file preview cards, two-step upload-then-submit flow, and richer results.

**Architecture:** Two new backend endpoints (`GET /ingest/collections/{name}`, `POST /synthesis/upload`); two rewritten Streamlit pages using existing `client`/`api_url` helpers. No changes to job store, evaluation, or agent code.

**Tech Stack:** FastAPI, Streamlit ≥1.28 (for `st.status`), httpx, qdrant-client, Python 3.11+, uv for running tests

---

## Task 1: Add `get_collection_info` to QdrantVectorStore

**Files:**
- Modify: `src/deps/qdrant_client.py`

**Step 1: Add the method**

Add to the `QdrantVectorStore` class after `list_collections`:

```python
def get_collection_info(self, collection_name: str) -> dict | None:
    """Return basic stats for a collection, or None if it doesn't exist."""
    if not self.client.collection_exists(collection_name):
        return None
    info = self.client.get_collection(collection_name)
    vectors_config = info.config.params.vectors
    if hasattr(vectors_config, "size"):
        dimensions = vectors_config.size
        distance = str(vectors_config.distance.value).lower()
    else:
        first = next(iter(vectors_config.values()))
        dimensions = first.size
        distance = str(first.distance.value).lower()
    return {
        "vectors_count": info.vectors_count or 0,
        "dimensions": dimensions,
        "distance": distance,
        "status": str(info.status.value).lower(),
    }
```

**Step 2: Commit**

```bash
git add src/deps/qdrant_client.py
git commit -m "feat: add get_collection_info to QdrantVectorStore"
```

---

## Task 2: Add `GET /ingest/collections/{name}` endpoint

**Files:**
- Modify: `src/api/routers/ingest.py`
- Test: `tests/api/test_ingest_router.py`

**Step 1: Write the failing test**

Append to `tests/api/test_ingest_router.py`:

```python
async def test_get_collection_info_found(client):
    mock_info = {
        "vectors_count": 42,
        "dimensions": 1536,
        "distance": "cosine",
        "status": "green",
    }
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.get_collection_info.return_value = mock_info
        response = await client.get("/api/v1/ingest/collections/research_papers")
    assert response.status_code == 200
    body = response.json()
    assert body["vectors_count"] == 42
    assert body["dimensions"] == 1536
    assert body["distance"] == "cosine"


async def test_get_collection_info_not_found(client):
    with patch("src.api.routers.ingest.QdrantVectorStore") as MockQdrant:
        MockQdrant.return_value.get_collection_info.return_value = None
        response = await client.get("/api/v1/ingest/collections/nonexistent")
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/api/test_ingest_router.py::test_get_collection_info_found -v
```
Expected: FAIL — `404 Not Found` (endpoint doesn't exist yet)

**Step 3: Add the endpoint to `src/api/routers/ingest.py`**

Add these imports at the top (after existing imports):
```python
from src.deps import QdrantVectorStore
from src.settings import settings
```

Add the endpoint after the existing `ingest_graph` function:

```python
@router.get("/collections/{name}")
async def get_collection_info(name: str):
    qs = QdrantVectorStore(uri=settings.qdrant_uri, api_key=settings.qdrant_api_key)
    info = qs.get_collection_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return info
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/api/test_ingest_router.py -v
```
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add src/api/routers/ingest.py tests/api/test_ingest_router.py
git commit -m "feat: add GET /ingest/collections/{name} endpoint for Qdrant stats"
```

---

## Task 3: Add `POST /synthesis/upload` endpoint

**Files:**
- Modify: `src/api/routers/synthesis.py`
- Test: `tests/api/test_synthesis_router.py`

**Step 1: Write the failing tests**

Append to `tests/api/test_synthesis_router.py`:

```python
async def test_upload_synthesis_files_success(client):
    response = await client.post(
        "/api/v1/synthesis/upload",
        files=[
            ("files", ("paper1.pdf", b"%PDF-1.4 content1", "application/pdf")),
            ("files", ("paper2.pdf", b"%PDF-1.4 content2", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["file_count"] == 2
    assert "file_dir" in body
    # Cleanup
    import shutil, os
    if os.path.isdir(body["file_dir"]):
        shutil.rmtree(body["file_dir"])


async def test_upload_synthesis_files_empty(client):
    response = await client.post("/api/v1/synthesis/upload", files=[])
    assert response.status_code == 422  # FastAPI validation: files required
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/api/test_synthesis_router.py::test_upload_synthesis_files_success -v
```
Expected: FAIL — `404 Not Found` (endpoint missing)

**Step 3: Add the endpoint to `src/api/routers/synthesis.py`**

Add imports at top (after existing imports):
```python
import os
import tempfile
```

Add the endpoint after the existing imports and before `_run_synthesis`:

```python
@router.post("/upload")
async def upload_synthesis_files(files: list[UploadFile] = File(...)):
    tmp_dir = tempfile.mkdtemp(prefix="synthesis_")
    for file in files:
        content = await file.read()
        filename = file.filename or f"upload_{uuid4().hex}.pdf"
        dest = os.path.join(tmp_dir, filename)
        with open(dest, "wb") as f:
            f.write(content)
    return {"file_dir": tmp_dir, "file_count": len(files)}
```

Also add the missing import at the top of the file:
```python
from uuid import uuid4
from fastapi import File, UploadFile
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/api/test_synthesis_router.py -v
```
Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/api/routers/synthesis.py tests/api/test_synthesis_router.py
git commit -m "feat: add POST /synthesis/upload endpoint for multi-file upload"
```

---

## Task 4: Rewrite `1_Ingestion.py` with enhanced UI

**Files:**
- Modify: `src/ui/pages/1_Ingestion.py`

**Step 1: Rewrite the file**

Replace the entire content of `src/ui/pages/1_Ingestion.py` with:

```python
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
```

**Step 2: Manually verify in browser**

```bash
# Terminal 1
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
uv run streamlit run src/ui/Home.py
```

Check:
- Navigate to Ingestion page
- Upload a PDF → preview card shows filename + size
- Click Ingest → `st.status` expands with 5 ✅ steps on success
- Result summary card shows correct metrics
- Collection Summary form works independently

**Step 3: Commit**

```bash
git add src/ui/pages/1_Ingestion.py
git commit -m "feat: enhance Ingestion page with file preview, status steps, result card, and collection summary"
```

---

## Task 5: Rewrite `4_Synthesis.py` with multi-file upload and enhanced UI

**Files:**
- Modify: `src/ui/pages/4_Synthesis.py`

**Step 1: Rewrite the file**

Replace the entire content of `src/ui/pages/4_Synthesis.py` with:

```python
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
                    f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
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
            st.success(
                f"✅ {upload_body['file_count']} file(s) uploaded to server"
            )
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
```

**Step 2: Manually verify in browser**

```bash
# Terminal 1
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
uv run streamlit run src/ui/Home.py
```

Check:
- Navigate to Synthesis page
- Upload 2–3 PDFs → preview cards appear in a 3-column grid with filename + size
- "Start Synthesis" is disabled until at least one file is uploaded
- Clicking Start shows "Uploading…" spinner, then "Submitting job…" spinner
- Polling shows pending/running status updates
- On completion, result metrics (goldens, files, avg) are displayed

**Step 3: Commit**

```bash
git add src/ui/pages/4_Synthesis.py
git commit -m "feat: enhance Synthesis page with multi-file upload, file preview cards, and richer results"
```

---

## Final Verification

Run the full test suite to confirm nothing is broken:

```bash
uv run pytest tests/api/test_ingest_router.py tests/api/test_synthesis_router.py -v
```

Expected: all tests pass.
