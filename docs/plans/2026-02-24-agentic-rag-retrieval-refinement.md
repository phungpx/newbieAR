# Agentic RAG & Retrieval Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs, improve code quality, and wire up cross-encoder reranking + score threshold filtering across `src/agents/` and `src/retrieval/`.

**Architecture:** Inline fixes within the existing file structure — no new files. Tools are registered via a side-effect import in `agentic_rag.py`. `BasicRAG.retrieve()` becomes async to support the async cross-encoder. `GraphRAG.retrieve()` is already async. Both tools in `tools.py` become direct `await` calls.

**Tech Stack:** `pydantic-ai`, `loguru`, `qdrant-client`, `graphiti-core`, `sentence-transformers` (for `SentenceTransformersReranker`), `pydantic-settings`

---

## Task 1: Fix — import tools.py in agentic_rag.py

**Problem:** `tools.py` decorates `agentic_rag` via `@agentic_rag.tool`, but is never imported anywhere. The agent has zero tools at runtime.

**Files:**
- Modify: `src/agents/agentic_rag.py`
- Create: `tests/agents/test_agentic_rag_tools.py`

**Step 1: Create tests directory structure**

```bash
mkdir -p tests/agents
touch tests/__init__.py tests/agents/__init__.py
```

**Step 2: Write the failing test**

Create `tests/agents/test_agentic_rag_tools.py`:

```python
from src.agents.agentic_rag import agentic_rag


def test_agentic_rag_has_tools_registered():
    tool_names = {t.name for t in agentic_rag.tools}
    assert "search_basic_rag" in tool_names
    assert "search_graphiti" in tool_names
```

**Step 3: Run test to verify it fails**

```bash
uv run pytest tests/agents/test_agentic_rag_tools.py -v
```

Expected: FAIL — tool_names will be empty `set()`.

**Step 4: Add the side-effect import to agentic_rag.py**

At the very end of `src/agents/agentic_rag.py`, after the agent definition (line 23), add:

```python
import src.agents.tools  # noqa: F401 — registers tools via @agentic_rag.tool decorators
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/agents/test_agentic_rag_tools.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/agentic_rag.py tests/agents/test_agentic_rag_tools.py tests/__init__.py tests/agents/__init__.py
git commit -m "fix: import tools.py so agent tools are registered at runtime"
```

---

## Task 2: Fix — search_graphiti: async misuse + wrong method call

**Problem 1:** `search_graphiti` wraps `ctx.deps.graph_rag.generate` in `asyncio.to_thread`. `generate` is `async` — wrapping async functions in `to_thread` does not await them; it runs the coroutine object in a thread without executing it.

**Problem 2:** It calls `generate()` (retrieval + LLM generation) instead of `retrieve()`. The agent does generation itself; the tool should only retrieve.

**Files:**
- Modify: `src/agents/tools.py`
- Test: `tests/agents/test_agentic_rag_tools.py`

**Step 1: Write a test verifying search_graphiti calls retrieve, not generate**

Add to `tests/agents/test_agentic_rag_tools.py`:

```python
import inspect
from src.agents import tools as tools_module
import src.agents.tools  # ensure import


def test_search_graphiti_calls_retrieve_not_generate():
    """Verify the tool implementation awaits retrieve(), not generate()."""
    source = inspect.getsource(tools_module.search_graphiti)
    assert "graph_rag.retrieve" in source
    assert "graph_rag.generate" not in source
    assert "asyncio.to_thread" not in source
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/agents/test_agentic_rag_tools.py::test_search_graphiti_calls_retrieve_not_generate -v
```

Expected: FAIL

**Step 3: Fix search_graphiti in tools.py**

In `src/agents/tools.py`, replace the body of `search_graphiti` (lines 93–95) from:

```python
        contexts, citations = await asyncio.to_thread(
            ctx.deps.graph_rag.generate, query, num_results=ctx.deps.top_k
        )
```

With:

```python
        contexts, citations = await ctx.deps.graph_rag.retrieve(query, top_k=ctx.deps.top_k)
```

Also remove the `asyncio` import if it is only used for `to_thread` in this function (check if `search_basic_rag` still uses it — it does, so keep the import).

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/agents/test_agentic_rag_tools.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/agents/tools.py tests/agents/test_agentic_rag_tools.py
git commit -m "fix: search_graphiti awaits retrieve() directly instead of asyncio.to_thread(generate)"
```

---

## Task 3: Code quality — remove global settings mutation from BasicRAG

**Problem:** `BasicRAG.__init__` writes `settings.qdrant_collection_name = qdrant_collection_name`, mutating the global singleton. Any code reading `settings.qdrant_collection_name` after this gets the last value written by any `BasicRAG` instance.

**Files:**
- Modify: `src/retrieval/basic_rag.py`
- Create: `tests/retrieval/test_basic_rag.py`

**Step 1: Create tests directory**

```bash
mkdir -p tests/retrieval
touch tests/retrieval/__init__.py
```

**Step 2: Write failing test**

Create `tests/retrieval/test_basic_rag.py`:

```python
from unittest.mock import MagicMock, patch
from src.settings import settings


def test_basic_rag_does_not_mutate_global_settings():
    original = settings.qdrant_collection_name

    with patch("src.retrieval.basic_rag.QdrantVectorStore"), \
         patch("src.retrieval.basic_rag.OpenAIEmbedding"), \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):
        from src.retrieval.basic_rag import BasicRAG
        BasicRAG(qdrant_collection_name="test_collection")

    assert settings.qdrant_collection_name == original, (
        "BasicRAG.__init__ must not mutate the global settings singleton"
    )


def test_basic_rag_uses_provided_collection_name():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding"), \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):
        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="my_collection")

    assert rag.collection_name == "my_collection"
```

**Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/retrieval/test_basic_rag.py -v
```

Expected: FAIL

**Step 4: Fix BasicRAG.__init__**

In `src/retrieval/basic_rag.py`, change `__init__`:

```python
def __init__(self, qdrant_collection_name: str = None):
    self.collection_name = qdrant_collection_name or settings.qdrant_collection_name
    self.vector_store = QdrantVectorStore(
        uri=settings.qdrant_uri,
        api_key=settings.qdrant_api_key,
    )
    self.embedder = OpenAIEmbedding(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model_id=settings.embedding_model,
    )
    self.llm = OpenAILLMClient(
        base_url=settings.llm_base_url,
        api_keys=settings.llm_api_key,
        model_id=settings.llm_model,
    )
    self.cross_encoder = None
```

Then in `retrieve()`, change `settings.qdrant_collection_name` to `self.collection_name`:

```python
retrieved_documents = self.vector_store.query(
    collection_name=self.collection_name,
    query_vector=embedding[0],
    top_k=top_k,
)
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/retrieval/test_basic_rag.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/retrieval/basic_rag.py tests/retrieval/test_basic_rag.py tests/retrieval/__init__.py
git commit -m "fix: BasicRAG stores collection_name as instance field, stops mutating global settings"
```

---

## Task 4: Code quality — fix GraphRAG.initialize_graphiti_client

**Problem:** `initialize_graphiti_client` creates `GraphitiClient()` (a second instance, discarding `self.graphiti_client`) instead of calling `self.graphiti_client.create_client(...)`.

**Files:**
- Modify: `src/retrieval/graph_rag.py`

**Step 1: Fix initialize_graphiti_client**

In `src/retrieval/graph_rag.py`, replace:

```python
async def initialize_graphiti_client(self):
    if self.graphiti is None:
        self.graphiti = await GraphitiClient().create_client(
            clear_existing_graphdb_data=False,
            max_coroutines=1,
        )
```

With:

```python
async def initialize_graphiti_client(self):
    if self.graphiti is None:
        self.graphiti = await self.graphiti_client.create_client(
            clear_existing_graphdb_data=False,
            max_coroutines=1,
        )
```

**Step 2: Verify no tests broken**

```bash
uv run pytest tests/ -v
```

Expected: all PASS (no new failures)

**Step 3: Commit**

```bash
git add src/retrieval/graph_rag.py
git commit -m "fix: initialize_graphiti_client uses self.graphiti_client instead of creating new instance"
```

---

## Task 5: Code quality — add reset() to AgentDependencies

**Problem:** Every turn in the agent loop manually nulls `deps.citations = None; deps.contexts = None`. This is caller boilerplate that belongs in the class.

**Files:**
- Modify: `src/agents/deps.py`
- Modify: `src/agents/agentic_rag.py` (update call site)
- Create: `tests/agents/test_deps.py`

**Step 1: Write failing test**

Create `tests/agents/test_deps.py`:

```python
from src.agents.deps import AgentDependencies


def test_reset_clears_citations_and_contexts():
    deps = AgentDependencies(
        citations=["source_a"],
        contexts=["some context"],
    )
    deps.reset()
    assert deps.citations is None
    assert deps.contexts is None


def test_reset_is_idempotent():
    deps = AgentDependencies()
    deps.reset()  # should not raise
    assert deps.citations is None
    assert deps.contexts is None
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_deps.py -v
```

Expected: FAIL — `AgentDependencies` has no `reset` method.

**Step 3: Add reset() to AgentDependencies**

Replace `src/agents/deps.py` with:

```python
from dataclasses import dataclass
from src.retrieval.graph_rag import GraphRAG
from src.retrieval.basic_rag import BasicRAG


@dataclass
class AgentDependencies:
    basic_rag: BasicRAG | None = None
    graph_rag: GraphRAG | None = None
    top_k: int = 5
    citations: list[str] | None = None
    contexts: list[str] | None = None

    def reset(self):
        self.citations = None
        self.contexts = None
```

**Step 4: Update the call site in agentic_rag.py**

In `src/agents/agentic_rag.py`, find (around line 107):

```python
            # Clean information in deps
            deps.citations = None
            deps.contexts = None
```

Replace with:

```python
            deps.reset()
```

**Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add src/agents/deps.py src/agents/agentic_rag.py tests/agents/test_deps.py
git commit -m "refactor: add AgentDependencies.reset() and use it in agent loop"
```

---

## Task 6: Update system prompt to describe both tools

**Problem:** `AGENTIC_RAG_INSTRUCTION` only mentions "vector-based document retrieval" and doesn't tell the agent it has two tools or when to use each.

**Files:**
- Modify: `src/prompts/agentic_rag_instruction.py`

**Step 1: Update the prompt**

Replace the entire content of `src/prompts/agentic_rag_instruction.py` with:

```python
AGENTIC_RAG_INSTRUCTION = """You are a helpful assistant with access to two complementary retrieval systems:

1. **search_basic_rag** — searches a vector database of document chunks. Best for finding specific passages, detailed explanations, and document-level content.
2. **search_graphiti** — searches a knowledge graph of facts, entities, and relationships. Best for factual lookups, entity relationships, and time-sensitive information.

When the user asks a question:
1. Choose the most appropriate tool (or both) based on the nature of the question.
2. Use the retrieved context to construct an accurate, grounded answer.
3. Cite sources when referencing specific documents or facts.
4. Be honest when the retrieved information is insufficient — do not hallucinate.

If results are insufficient:
- Try rephrasing or decomposing the query.
- Ask the user for clarification.
- Acknowledge the limitations of the available knowledge.

Always prioritize accuracy over completeness."""
```

**Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

**Step 3: Commit**

```bash
git add src/prompts/agentic_rag_instruction.py
git commit -m "docs: update agentic RAG system prompt to describe both retrieval tools"
```

---

## Task 7: Add score_threshold setting + make BasicRAG.retrieve async

**Context:** Cross-encoder reranking (Task 8) is async. Making `retrieve()` async unifies the interface and lets `search_basic_rag` tool `await` it directly (same pattern as `search_graphiti`).

**Files:**
- Modify: `src/settings.py`
- Modify: `src/retrieval/basic_rag.py`
- Modify: `src/agents/tools.py`
- Test: `tests/retrieval/test_basic_rag.py`

**Step 1: Add retrieval_score_threshold to QdrantVectorStoreSettings**

In `src/settings.py`, update `QdrantVectorStoreSettings`:

```python
class QdrantVectorStoreSettings(ProjectBaseSettings):
    qdrant_uri: str
    qdrant_api_key: str | None = None
    qdrant_collection_name: str
    retrieval_score_threshold: float = 0.0
```

**Step 2: Write failing tests for score threshold**

Add to `tests/retrieval/test_basic_rag.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.models import RetrievalInfo


@pytest.mark.asyncio
async def test_retrieve_filters_below_score_threshold():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point_low = MagicMock()
        mock_point_low.payload = {"text": "low score doc", "filename": "a.txt", "chunk_id": "1"}
        mock_point_low.score = 0.2

        mock_point_high = MagicMock()
        mock_point_high.payload = {"text": "high score doc", "filename": "b.txt", "chunk_id": "2"}
        mock_point_high.score = 0.8

        mock_result = MagicMock()
        mock_result.points = [mock_point_low, mock_point_high]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        results = await rag.retrieve("test query", top_k=5, score_threshold=0.5)

    assert len(results) == 1
    assert results[0].content == "high score doc"


@pytest.mark.asyncio
async def test_retrieve_returns_all_when_threshold_is_zero():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point = MagicMock()
        mock_point.payload = {"text": "doc", "filename": "a.txt", "chunk_id": "1"}
        mock_point.score = 0.1

        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        results = await rag.retrieve("test query", top_k=5, score_threshold=0.0)

    assert len(results) == 1
```

**Step 3: Run to verify they fail**

```bash
uv run pytest tests/retrieval/test_basic_rag.py -v -k "threshold"
```

Expected: FAIL — `retrieve` is sync, not async; no `score_threshold` param.

**Step 4: Make BasicRAG.retrieve async with score_threshold**

In `src/retrieval/basic_rag.py`, update `retrieve`:

```python
async def retrieve(
    self, query: str, top_k: int = 5, score_threshold: float = 0.0
) -> list[RetrievalInfo]:
    import asyncio

    embedding = await asyncio.to_thread(self.embedder.embed_texts, [query])
    retrieved_documents = await asyncio.to_thread(
        self.vector_store.query,
        collection_name=self.collection_name,
        query_vector=embedding[0],
        top_k=top_k,
    )

    retrieval_infos: list[RetrievalInfo] = []
    for doc in retrieved_documents.points:
        content = doc.payload.get("text", "")
        score = getattr(doc, "score", 0.0)
        filename = doc.payload.get("filename", "Unknown")
        chunk_id = doc.payload.get("chunk_id", "Unknown")
        source = f"{filename} - Chunk #{chunk_id}"

        retrieval_infos.append(
            RetrievalInfo(content=content, source=source, score=score)
        )

    if score_threshold > 0.0:
        retrieval_infos = [r for r in retrieval_infos if r.score >= score_threshold]

    return retrieval_infos
```

Also update `generate()` to `await self.retrieve(...)`:

```python
async def generate(
    self,
    query: str,
    top_k: int = 5,
    return_context: bool = False,
) -> tuple[list[RetrievalInfo], str] | str:
    retrieval_infos = await self.retrieve(query, top_k)
    # ... rest unchanged
```

And update the `__main__` block to use `asyncio.run`:

```python
if __name__ == "__main__":
    import asyncio
    import argparse
    from rich.panel import Panel
    from rich.console import Console

    async def _main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--qdrant_collection_name", type=str, required=True)
        parser.add_argument("--top_k", type=int, default=10)
        args = parser.parse_args()

        basic_rag = BasicRAG(qdrant_collection_name=args.qdrant_collection_name)
        console = Console()
        console.print(Panel.fit(
            f"Basic Retrieval CLI Mode - Collection: {args.qdrant_collection_name}",
            style="bold cyan",
        ))

        while True:
            try:
                query = console.input("[bold yellow]Enter a question (or 'exit'): [/bold yellow]")
                if query.lower() in ["exit", "quit"]:
                    break
                retrieval_infos, response = await basic_rag.generate(query, top_k=args.top_k, return_context=True)
                contexts = [r.content for r in retrieval_infos]
                citations = [r.source for r in retrieval_infos]
                display_rag_results(console, query, contexts, citations, response)
            except KeyboardInterrupt:
                break

    asyncio.run(_main())
```

**Step 5: Update search_basic_rag in tools.py**

In `src/agents/tools.py`, replace `search_basic_rag`'s retrieve call:

```python
    retrieval_infos = await ctx.deps.basic_rag.retrieve(query, top_k=ctx.deps.top_k)
```

Remove the `asyncio.to_thread` wrapper. Keep `import asyncio` for the timeout/connection error handling pattern (it's still used for exception types).

**Step 6: Install pytest-asyncio if needed**

```bash
uv add --dev pytest-asyncio
```

Add `pytest.ini` or `pyproject.toml` marker (check if `pyproject.toml` has `[tool.pytest.ini_options]`):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**Step 7: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

**Step 8: Commit**

```bash
git add src/settings.py src/retrieval/basic_rag.py src/agents/tools.py tests/retrieval/test_basic_rag.py
git commit -m "feat: make BasicRAG.retrieve async, add score_threshold filtering, update tools"
```

---

## Task 8: Wire cross-encoder reranking into BasicRAG.retrieve

**Context:** `SentenceTransformersReranker` is already implemented at `src/deps/cross_encoder/sentence_transformers_reranker.py`. Its `rank(query, passages)` method is async and returns `list[tuple[str, float]]` sorted descending by score. `BasicRAG` already has `self.cross_encoder = None`.

**Files:**
- Modify: `src/retrieval/basic_rag.py`
- Test: `tests/retrieval/test_basic_rag.py`

**Step 1: Write failing test for cross-encoder reranking**

Add to `tests/retrieval/test_basic_rag.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_reranks_when_cross_encoder_set():
    with patch("src.retrieval.basic_rag.QdrantVectorStore") as mock_store_cls, \
         patch("src.retrieval.basic_rag.OpenAIEmbedding") as mock_embed_cls, \
         patch("src.retrieval.basic_rag.OpenAILLMClient"):

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed_cls.return_value = mock_embedder

        mock_point_a = MagicMock()
        mock_point_a.payload = {"text": "doc A", "filename": "a.txt", "chunk_id": "1"}
        mock_point_a.score = 0.9  # high vector score

        mock_point_b = MagicMock()
        mock_point_b.payload = {"text": "doc B", "filename": "b.txt", "chunk_id": "2"}
        mock_point_b.score = 0.5  # lower vector score

        mock_result = MagicMock()
        mock_result.points = [mock_point_a, mock_point_b]
        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_cls.return_value = mock_store

        from src.retrieval.basic_rag import BasicRAG
        rag = BasicRAG(qdrant_collection_name="test")

        # Cross-encoder ranks doc B higher than doc A
        mock_cross_encoder = AsyncMock()
        mock_cross_encoder.rank.return_value = [("doc B", 0.95), ("doc A", 0.3)]
        rag.cross_encoder = mock_cross_encoder

        results = await rag.retrieve("test query", top_k=5)

    assert results[0].content == "doc B"
    assert results[1].content == "doc A"
    assert results[0].score == pytest.approx(0.95)
    assert results[1].score == pytest.approx(0.3)
```

**Step 2: Run to verify it fails**

```bash
uv run pytest tests/retrieval/test_basic_rag.py::test_retrieve_reranks_when_cross_encoder_set -v
```

Expected: FAIL

**Step 3: Add reranking to BasicRAG.retrieve**

After the score threshold filter in `retrieve()`, add:

```python
    if self.cross_encoder is not None and retrieval_infos:
        passages = [r.content for r in retrieval_infos]
        ranked = await self.cross_encoder.rank(query, passages)
        # ranked is list[tuple[str, float]] sorted descending by cross-encoder score
        score_map = {passage: score for passage, score in ranked}
        # Update scores and re-sort
        for r in retrieval_infos:
            r.score = score_map.get(r.content, r.score)
        retrieval_infos.sort(key=lambda r: r.score, reverse=True)

    return retrieval_infos
```

**Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/retrieval/basic_rag.py tests/retrieval/test_basic_rag.py
git commit -m "feat: wire cross-encoder reranking into BasicRAG.retrieve when cross_encoder is set"
```

---

## Final Verification

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests PASS, no warnings about unregistered tools or async misuse.
