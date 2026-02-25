# Design: Docling Paragraph-Level Chunking with RAG Context Construction

**Date:** 2026-02-25
**Branch:** features/update_synthesis
**Status:** Approved

## Problem

`synthesize.py` uses deepeval's built-in `generate_goldens_from_docs()` with `ContextConstructionConfig` (fixed `chunk_size=1024`, `chunk_overlap=20`). This naive character-count chunking splits text at arbitrary boundaries and constructs contexts from adjacent chunks only — degrading context quality for synthetic golden generation on structured documents like research papers.

## Goal

Replace deepeval's internal chunking + context construction with a two-phase pipeline:

1. **Pre-chunk** with docling's `HierarchicalChunker` (paragraph-aware, structure-respecting)
2. **Build contexts via retrieval**: embed all chunks, store in Qdrant, then for each context randomly pick a seed chunk and retrieve its k-1 most semantically similar neighbors

This produces semantically coherent, non-sequential contexts that better reflect real RAG retrieval scenarios.

## Architecture

### Before

```
file_path
  → deepeval internal loader (character-split, chunk_size=1024)
  → generate_goldens_from_docs()
  → goldens
```

### After

```
file_path
  → DocumentChunker(strategy='hierarchical')    # docling paragraph chunks
  → OpenAIEmbedding.embed_texts(all chunks)     # embed all at once
  → QdrantVectorStore.create_collection(        # temp collection per doc
        'synthesis_{file_stem}')
  → QdrantVectorStore.add_embeddings(...)       # store all chunks + vectors
  → repeat M=5 times:
       seed_idx = random.randint(0, N-1)
       query Qdrant with vectors[seed_idx], top_k=k+1
       neighbors = results excluding seed [:k-1]
       context = [seed.text] + [n.payload["text"] for n in neighbors]
  → QdrantVectorStore.delete_collection(...)    # cleanup
  → generate_goldens_from_contexts(contexts, source_files)
  → goldens
```

## Parameters

| Parameter | Value | Notes |
|---|---|---|
| `context_size` k | 3 | 1 seed + 2 neighbors |
| `num_contexts` M | 5 | contexts per document |
| query `top_k` | k+1 = 4 | fetch extra to filter out seed |
| collection name | `synthesis_{file_stem}` | unique per doc |

## Components

### 1. `src/synthesis/utils.py` — replace `build_contexts_from_doc`

**New signature:**

```python
def build_contexts_from_doc(
    file_path: str,
    embedder: OpenAIEmbedding,
    vector_store: QdrantVectorStore,
    embedding_size: int,
    num_contexts: int = 5,
    context_size: int = 3,
) -> list[list[str]]:
```

**Algorithm:**

```
1. chunker = DocumentChunker(strategy="hierarchical")
2. chunks, _ = chunker.chunk_document(file_path)
3. texts = [c.text for c in chunks]
4. if not texts: return []
5. vectors = embedder.embed_texts(texts)          # N vectors
6. collection = f"synthesis_{Path(file_path).stem}"
7. vector_store.create_collection(collection, embedding_size)
8. vector_store.add_embeddings(
       collection,
       embeddings=vectors,
       payloads=[{"text": t, "chunk_idx": i} for i, t in enumerate(texts)],
       ids=list(range(len(texts))),
   )
9. contexts = []
   for _ in range(min(num_contexts, len(texts))):
       seed_idx = random.randint(0, len(texts) - 1)
       seed_vec = vectors[seed_idx]
       results = vector_store.query(collection, seed_vec, top_k=context_size + 1)
       neighbors = [
           r for r in results.points
           if r.payload["chunk_idx"] != seed_idx
       ][:context_size - 1]
       context = [texts[seed_idx]] + [n.payload["text"] for n in neighbors]
       contexts.append(context)
10. vector_store.delete_collection(collection)
11. return contexts
```

**Imports added to `utils.py`:**

```python
import random
from pathlib import Path
from src.deps import DocumentChunker, OpenAIEmbedding, QdrantVectorStore
```

### 2. `src/synthesis/synthesize.py` — module level + `__main__`

**Add at module level** (alongside existing `model = GPTModel(...)`, which is already module-level):

```python
from src.deps import OpenAIEmbedding, QdrantVectorStore
from src.synthesis.utils import save_goldens_to_files, build_contexts_from_doc

embedder = OpenAIEmbedding(
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
    model_id=settings.embedding_model,
)
vector_store = QdrantVectorStore(
    uri=settings.qdrant_uri,
    api_key=settings.qdrant_api_key,
)
```

**In `__main__` loop — replace synthesis call:**

```python
for file_path in file_paths:
    logger.info(f"Synthesizing {file_path}")
    contexts = build_contexts_from_doc(
        str(file_path),
        embedder=embedder,
        vector_store=vector_store,
        embedding_size=settings.embedding_dimensions,
        num_contexts=5,
        context_size=3,
    )
    logger.info(f"Built {len(contexts)} contexts from {file_path.name}")
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1,
        source_files=[str(file_path)] * len(contexts),
    )
    logger.info(f"Synthesis cost: {synthesizer.synthesis_cost}")
    save_goldens_to_files(goldens, output_dir)
```

**Remove:**
- `from deepeval.models.embedding_models import LocalEmbeddingModel`
- `from deepeval.synthesizer.config import ContextConstructionConfig` (from the import block)
- `embeder = LocalEmbeddingModel(...)` variable
- `context_construction_config = ContextConstructionConfig(...)` variable

**Keep unchanged:** `FiltrationConfig`, `EvolutionConfig`, `StylingConfig`, `Synthesizer`, `STYLING_CONFIG`, `TOPIC`, argparse setup.

## Data Flow

```
ChunkInfo list (N paragraphs)
  ↓  embed all at once
list[list[float]] (N vectors)
  ↓  store in Qdrant 'synthesis_{stem}' with {text, chunk_idx} payloads
  ↓  for each of M=5 contexts:
       pick random seed → query k+1 neighbors → filter seed → take k-1
       context = [seed_text, neighbor1_text, neighbor2_text]
list[list[str]] (5 contexts × 3 strings each)
  ↓  generate_goldens_from_contexts(source_files=[file_path]*5)
list[Golden]
  ↓  save_goldens_to_files()
JSON files in output_dir/
```

## Edge Cases

| Case | Behavior |
|---|---|
| N < context_size (doc has fewer chunks than k) | neighbors list is shorter; context has < k strings. Still valid. |
| N = 0 (empty doc / parse failure) | `build_contexts_from_doc` returns `[]`; loop is skipped. |
| num_contexts > N (request more contexts than chunks) | `min(num_contexts, len(texts))` caps the loop. |
| Qdrant create/add fails | Exception propagates; no goldens saved for this doc. Logged by caller. |
| Qdrant delete fails | Exception propagates after synthesis. Consider try/finally for cleanup. |

## Testing

Unit tests in `tests/synthesis/test_utils.py`:
- Mock `DocumentChunker`, `OpenAIEmbedding`, `QdrantVectorStore`
- Test normal case (N=5, k=3, M=5) → 5 contexts each of length ≤ 3
- Test empty document → `[]`
- Test N < k (2 chunks, k=3) → 1 context with 2 strings
- Test seed is excluded from neighbors
- Test collection is always deleted (even on error → try/finally)

## Files Changed

- `src/synthesis/utils.py` — replace `build_contexts_from_doc` with RAG-based version
- `src/synthesis/synthesize.py` — add `embedder`/`vector_store` at module level, remove `embeder`/`ContextConstructionConfig`, replace synthesis call
