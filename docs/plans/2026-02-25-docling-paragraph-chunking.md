# Docling RAG-Based Context Construction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace deepeval's internal chunking in `synthesize.py` with a two-phase pipeline: docling hierarchical chunking → embed all chunks → store in Qdrant → build contexts by picking random seed chunks and retrieving their k-1 nearest neighbors.

**Architecture:** `build_contexts_from_doc()` in `utils.py` accepts injected `OpenAIEmbedding` and `QdrantVectorStore` deps, chunks the document with `DocumentChunker(strategy="hierarchical")`, embeds all chunks in one batch, stores them in a temporary Qdrant collection, then samples `num_contexts` seed chunks and retrieves their `context_size-1` nearest neighbors each. The temp collection is always cleaned up via `try/finally`. `synthesize.py` instantiates the deps at module level from `settings` and replaces `generate_goldens_from_docs()` with the new pipeline.

**Tech Stack:** docling `HierarchicalChunker` (via `DocumentChunker`), `OpenAIEmbedding`, `QdrantVectorStore` (from `src/deps`), deepeval `Synthesizer.generate_goldens_from_contexts()`, pytest + `unittest.mock`

---

### Task 1: Write failing tests for `build_contexts_from_doc`

**Files:**
- Create: `tests/synthesis/__init__.py`
- Create: `tests/synthesis/test_utils.py`

**Step 1: Create the test package**

```bash
touch tests/synthesis/__init__.py
```

**Step 2: Write all failing tests**

Create `tests/synthesis/test_utils.py` with this exact content:

```python
import pytest
from unittest.mock import MagicMock, patch

from src.synthesis.utils import build_contexts_from_doc


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_point(chunk_idx: int, text: str):
    point = MagicMock()
    point.payload = {"text": text, "chunk_idx": chunk_idx}
    return point


def _make_query_result(*points):
    result = MagicMock()
    result.points = list(points)
    return result


def _patch_chunker(texts: list[str]):
    """Patch DocumentChunker so chunk_document() returns chunks with given texts."""
    chunks = [MagicMock(text=t) for t in texts]
    mock_instance = MagicMock()
    mock_instance.chunk_document.return_value = (chunks, None)
    return patch("src.synthesis.utils.DocumentChunker", return_value=mock_instance)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def embedder():
    m = MagicMock()
    m.embed_texts.return_value = [[float(i), 0.0] for i in range(10)]
    return m


@pytest.fixture
def vector_store():
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_document_returns_empty_list(embedder, vector_store):
    with _patch_chunker([]):
        result = build_contexts_from_doc(
            "/fake/doc.pdf", embedder, vector_store, embedding_size=2
        )
    assert result == []
    embedder.embed_texts.assert_not_called()
    vector_store.create_collection.assert_not_called()
    vector_store.delete_collection.assert_not_called()


def test_normal_case_returns_num_contexts_contexts(embedder, vector_store):
    texts = [f"para {i}" for i in range(5)]
    embedder.embed_texts.return_value = [[float(i), 0.0] for i in range(5)]
    # query returns 4 points; seed_idx=1 should be excluded from neighbors
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "para 0"),
        _make_point(1, "para 1"),
        _make_point(2, "para 2"),
        _make_point(3, "para 3"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=1):
            result = build_contexts_from_doc(
                "/fake/doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=3, context_size=3,
            )
    assert len(result) == 3
    # seed_idx=1 → neighbors=[chunk_idx 0, chunk_idx 2] → context length = 3
    assert result[0][0] == "para 1"
    assert len(result[0]) == 3


def test_seed_is_excluded_from_neighbors(embedder, vector_store):
    texts = ["s0", "s1", "s2", "s3", "s4"]
    embedder.embed_texts.return_value = [[float(i), 0.0] for i in range(5)]
    # query returns seed (idx=2) as first result — it must be filtered out
    vector_store.query.return_value = _make_query_result(
        _make_point(2, "s2"),  # ← seed, must be excluded
        _make_point(0, "s0"),
        _make_point(4, "s4"),
        _make_point(3, "s3"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=2):
            result = build_contexts_from_doc(
                "/fake/doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=1, context_size=3,
            )
    context = result[0]
    assert context[0] == "s2"       # seed is first
    assert "s2" not in context[1:]  # seed NOT repeated as a neighbor


def test_collection_deleted_on_success(embedder, vector_store):
    texts = ["p0", "p1", "p2"]
    embedder.embed_texts.return_value = [[0.0, 0.0]] * 3
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "p0"), _make_point(1, "p1"), _make_point(2, "p2"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=0):
            build_contexts_from_doc(
                "/fake/my_doc.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=1,
            )
    vector_store.delete_collection.assert_called_once_with("synthesis_my_doc")


def test_collection_deleted_on_error(embedder, vector_store):
    texts = ["p0", "p1"]
    embedder.embed_texts.return_value = [[0.0, 0.0]] * 2
    vector_store.query.side_effect = RuntimeError("Qdrant down")
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", return_value=0):
            with pytest.raises(RuntimeError, match="Qdrant down"):
                build_contexts_from_doc(
                    "/fake/err_doc.pdf", embedder, vector_store,
                    embedding_size=2, num_contexts=1,
                )
    # delete_collection must still be called even after the error
    vector_store.delete_collection.assert_called_once_with("synthesis_err_doc")


def test_uses_hierarchical_chunker_strategy(embedder, vector_store):
    with patch("src.synthesis.utils.DocumentChunker") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.chunk_document.return_value = ([], None)
        mock_cls.return_value = mock_instance
        build_contexts_from_doc("/fake/doc.pdf", embedder, vector_store, embedding_size=2)
        mock_cls.assert_called_once_with(strategy="hierarchical")


def test_num_contexts_capped_by_chunk_count(embedder, vector_store):
    """num_contexts > N should produce only N contexts."""
    texts = ["only", "two"]
    embedder.embed_texts.return_value = [[0.0, 0.0], [1.0, 0.0]]
    vector_store.query.return_value = _make_query_result(
        _make_point(0, "only"), _make_point(1, "two"),
    )
    with _patch_chunker(texts):
        with patch("src.synthesis.utils.random.randint", side_effect=[0, 1]):
            result = build_contexts_from_doc(
                "/fake/small.pdf", embedder, vector_store,
                embedding_size=2, num_contexts=10, context_size=3,
            )
    assert len(result) == 2  # capped at min(10, 2) = 2
```

**Step 3: Run to confirm all tests fail**

```bash
uv run pytest tests/synthesis/test_utils.py -v
```

Expected: `ImportError` — `build_contexts_from_doc` does not yet accept these parameters (or doesn't exist in the new form). All 7 tests FAIL.

---

### Task 2: Implement `build_contexts_from_doc` in `utils.py`

**Files:**
- Modify: `src/synthesis/utils.py`

**Step 1: Replace the file contents**

Open `src/synthesis/utils.py` and replace it entirely with:

```python
import json
import random
from pathlib import Path
from loguru import logger
from uuid import uuid4
from deepeval.dataset.golden import Golden

from src.deps import DocumentChunker, OpenAIEmbedding, QdrantVectorStore


def save_goldens_to_files(goldens: list[Golden], output_dir: str = "goldens"):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

    for golden in goldens:
        file_dir = output_dir / Path(golden.source_file).stem
        if not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {file_dir}")

        file_path = file_dir / f"{uuid4()}.json"
        golden_data = golden.model_dump(by_alias=True, exclude_none=True)

        try:
            with file_path.open(mode="w", encoding="utf-8") as f:
                json.dump(golden_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    logger.info(f"\nSuccessfully saved {len(goldens)} files to '{output_dir}'.")


def build_contexts_from_doc(
    file_path: str,
    embedder: OpenAIEmbedding,
    vector_store: QdrantVectorStore,
    embedding_size: int,
    num_contexts: int = 5,
    context_size: int = 3,
) -> list[list[str]]:
    """Chunk a document with docling, embed all chunks, store in Qdrant, then
    build semantically coherent contexts by retrieving k-1 nearest neighbors
    for randomly selected seed chunks.

    Args:
        file_path: Path to the document (PDF, DOCX, etc.)
        embedder: OpenAIEmbedding instance for embedding chunk texts.
        vector_store: QdrantVectorStore instance for similarity search.
        embedding_size: Dimensionality of the embedding vectors.
        num_contexts: Number of contexts to build per document.
        context_size: Chunks per context (1 seed + context_size-1 neighbors).

    Returns:
        List of contexts, each a list of paragraph strings.
        Returns [] if the document produces no chunks.
    """
    chunker = DocumentChunker(strategy="hierarchical")
    chunks, _ = chunker.chunk_document(file_path)
    texts = [c.text for c in chunks]

    if not texts:
        logger.warning(f"No chunks extracted from {file_path}. Skipping.")
        return []

    vectors = embedder.embed_texts(texts)
    collection_name = f"synthesis_{Path(file_path).stem}"

    try:
        vector_store.create_collection(collection_name, embedding_size)
        vector_store.add_embeddings(
            collection_name,
            embeddings=vectors,
            payloads=[{"text": t, "chunk_idx": i} for i, t in enumerate(texts)],
            ids=list(range(len(texts))),
        )

        contexts = []
        for _ in range(min(num_contexts, len(texts))):
            seed_idx = random.randint(0, len(texts) - 1)
            seed_vec = vectors[seed_idx]
            results = vector_store.query(
                collection_name, seed_vec, top_k=context_size + 1
            )
            neighbors = [
                r for r in results.points
                if r.payload["chunk_idx"] != seed_idx
            ][:context_size - 1]
            context = [texts[seed_idx]] + [n.payload["text"] for n in neighbors]
            contexts.append(context)

        return contexts
    finally:
        vector_store.delete_collection(collection_name)
```

**Step 2: Run tests to confirm they pass**

```bash
uv run pytest tests/synthesis/test_utils.py -v
```

Expected: All 7 tests PASS.

**Step 3: Commit**

```bash
git add tests/synthesis/__init__.py tests/synthesis/test_utils.py src/synthesis/utils.py
git commit -m "[Updated] build_contexts_from_doc: embed→store→retrieve RAG context construction"
```

---

### Task 3: Refactor `synthesize.py`

**Files:**
- Modify: `src/synthesis/synthesize.py`

**Step 1: Update the imports block**

The current imports block (top of file) looks like this:

```python
from enum import Enum
from pathlib import Path
from loguru import logger

# from src.synthesis.bedrock_model import AmazonBedrockModel
from deepeval.models.llms import GPTModel
from deepeval.models.embedding_models import LocalEmbeddingModel
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from dataclasses import dataclass
from deepeval.synthesizer import Synthesizer, Evolution

from src.settings import settings

# from src.evals.bedrock_llm_wrapper import BedrockLLMWrapper
from src.synthesis.utils import save_goldens_to_files
```

Replace it with:

```python
from enum import Enum
from pathlib import Path
from loguru import logger

# from src.synthesis.bedrock_model import AmazonBedrockModel
from deepeval.models.llms import GPTModel
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
)
from dataclasses import dataclass
from deepeval.synthesizer import Synthesizer, Evolution

from src.settings import settings
from src.deps import OpenAIEmbedding, QdrantVectorStore

# from src.evals.bedrock_llm_wrapper import BedrockLLMWrapper
from src.synthesis.utils import save_goldens_to_files, build_contexts_from_doc
```

**Step 2: Remove the `embeder` block**

Find and delete these lines (around line 79–83):

```python
embeder = LocalEmbeddingModel(
    model=settings.embedding_model,
    base_url=settings.embedding_base_url,
    api_key=settings.embedding_api_key,
)
```

**Step 3: Remove the `context_construction_config` block**

Find and delete these lines (around line 115–126, including the comment):

```python
# Settings for building RAG
context_construction_config = ContextConstructionConfig(
    embedder=embeder,
    critic_model=model,
    encoding="utf-8",
    chunk_size=1024,
    chunk_overlap=20,
    max_contexts_per_document=5,
    min_contexts_per_document=3,
    max_context_length=5,
    min_context_length=3,
)
```

**Step 4: Add `embedder` and `vector_store` at module level**

After the existing `synthesizer = Synthesizer(...)` block, add:

```python
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

**Step 5: Replace the synthesis call in `__main__`**

Find this block inside the `for file_path in file_paths:` loop:

```python
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(file_path)],
        include_expected_output=True,
        context_construction_config=context_construction_config,
        max_goldens_per_context=1,
    )
```

Replace it with:

```python
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
```

**Step 6: Run tests to confirm no regressions**

```bash
uv run pytest tests/synthesis/test_utils.py -v
```

Expected: All 7 tests PASS.

**Step 7: Commit**

```bash
git add src/synthesis/synthesize.py
git commit -m "[Updated] synthesize.py: RAG-based context construction via docling + Qdrant"
```

---

### Task 4: Final verification

**Files:**
- Read: `src/synthesis/synthesize.py`
- Read: `src/synthesis/utils.py`

**Step 1: Verify `synthesize.py` has no leftover references**

Check that these no longer appear anywhere in the file:
- `LocalEmbeddingModel`
- `ContextConstructionConfig`
- `embeder` (the old misspelled variable — the new one is `embedder`)
- `context_construction_config`
- `generate_goldens_from_docs`

**Step 2: Verify `synthesize.py` has all required pieces**

Check that these are present:
- `from src.deps import OpenAIEmbedding, QdrantVectorStore`
- `from src.synthesis.utils import save_goldens_to_files, build_contexts_from_doc`
- `embedder = OpenAIEmbedding(...)` at module level
- `vector_store = QdrantVectorStore(...)` at module level
- `build_contexts_from_doc(...)` in the `__main__` loop
- `generate_goldens_from_contexts(...)` in the `__main__` loop

**Step 3: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass. Zero failures.
