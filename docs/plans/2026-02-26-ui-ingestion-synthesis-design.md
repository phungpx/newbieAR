# UI Enhancement Design: Ingestion & Synthesis Pages

**Date:** 2026-02-26
**Branch:** features/update_ui
**Scope:** `src/ui/pages/1_Ingestion.py`, `src/ui/pages/4_Synthesis.py`, `src/api/routers/ingest.py`, `src/api/routers/synthesis.py`

---

## 1. Ingestion Page (`1_Ingestion.py`)

### Goals
- Show a file preview card (name, size) before submitting
- Animate step-by-step processing status using `st.status` (UI only — backend stays synchronous)
- Display a styled result summary card after completion
- Add a standalone "Collection Summary" panel that queries live Qdrant stats

### Layout

```
[ Vector DB ] [ Graph DB ]       ← radio

[ Upload PDF — Browse files ]
📄 paper.pdf  |  1.2 MB  |  PDF  ← file preview card (appears on select)

Collection name: [research_papers]
Chunk strategy: [hybrid ▼]

[ Ingest ]

── Processing Status ──
▼ Ingesting paper.pdf...
  ✓ File uploaded
  ✓ PDF parsed
  ✓ Document chunked
  ✓ Chunks embedded
  ✓ Saved to vector store

── Result Summary ──
Collection: research_papers  |  Strategy: hybrid
Doc path: data/papers/docs/paper.md
Chunk path: data/papers/chunks/paper.json

── Collection Summary ──
Collection name: [research_papers]  [View Summary]
Vectors: 142  |  Dimensions: 1536  |  Distance: cosine
```

### Backend Change
New endpoint: `GET /api/v1/ingest/collections/{name}`
Queries Qdrant via `QdrantVectorStore` and returns:
```json
{ "vectors_count": 142, "dimensions": 1536, "distance": "cosine", "status": "green" }
```

### Step Simulation Strategy
Since ingestion is synchronous, all 5 steps are shown as "in progress" while the single API call runs, then all marked done (or error) on response. Uses `st.status` with `expanded=True` during processing, collapses to a success summary after.

---

## 2. Synthesis Page (`4_Synthesis.py`)

### Goals
- Replace directory path input with multi-file uploader
- Show a metadata card per uploaded file (name + size)
- Two-step submit: upload files first, then trigger synthesis job
- Enhanced result display: goldens count, files processed, average per file

### Layout

```
[ Upload PDFs — Browse files (multiple) ]

── Uploaded Files (3) ──
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 📄 paper1.pdf│ │ 📄 paper2.pdf│ │ 📄 paper3.pdf│
│  1.2 MB      │ │  890 KB      │ │  2.1 MB      │
└──────────────┘ └──────────────┘ └──────────────┘

Output dir: [data/goldens]    Topic: [paper ▼]
                              Contexts: [5 ────]
                              Context size: [5 ────]

[ Start Synthesis ]

── Processing Status ──
Job ID: `abc-123`
Status: ● running

── Results (on done) ──
✓ 42 goldens saved to data/goldens
Goldens: 42  |  Files: 3  |  Avg: 14/file
```

### Backend Change
New endpoint: `POST /api/v1/synthesis/upload`
Accepts `files: list[UploadFile]`, saves each to a server-side temp directory, returns:
```json
{ "file_dir": "/tmp/synthesis_abc123/", "file_count": 3 }
```

The existing `POST /synthesis/jobs` endpoint is unchanged — it receives `file_dir` pointing to the temp directory.

### Two-Step Flow
1. UI calls `POST /synthesis/upload` with all files → gets `tmp_dir`
2. UI calls `POST /synthesis/jobs` with `file_dir=tmp_dir` + other params → gets `job_id`
3. UI polls `GET /synthesis/jobs/{job_id}` every 3s until `done` or `failed`

---

## 3. No Changes Required
- `src/api/job_store.py` — unchanged
- `src/api/routers/evaluation.py` — unchanged
- `src/ui/api_client.py` — unchanged
- Synthesis job background logic — unchanged (already reads from `file_dir`)
