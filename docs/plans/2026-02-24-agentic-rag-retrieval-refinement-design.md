# Design: Agentic RAG & Retrieval Refinement

**Date:** 2026-02-24
**Branch:** `features/improve_agents`
**Approach:** Inline fix + minimal restructure (no new files)

---

## Problem Statement

Several bugs and design issues exist across `src/agents/` and `src/retrieval/` that degrade correctness and maintainability:

1. `tools.py` is never imported — the agent has zero tools registered at runtime.
2. `search_graphiti` wraps an `async` method in `asyncio.to_thread`, which is incorrect.
3. `search_graphiti` calls `generate()` (retrieval + LLM) instead of `retrieve()` (retrieval only).
4. `BasicRAG.__init__` mutates the global `settings.qdrant_collection_name` singleton.
5. `GraphRAG.initialize_graphiti_client` creates a new `GraphitiClient()` instead of using `self.graphiti_client`.
6. `AgentDependencies` `citations`/`contexts` reset is left to callers as boilerplate.
7. System prompt only mentions vector retrieval, not the graph retrieval tool.
8. `BasicRAG.retrieve()` has a wired-but-unused `cross_encoder` field with no reranking or threshold logic.

---

## Design

### Bug Fixes

#### 1. Import `tools.py` in `agentic_rag.py`

At the bottom of `src/agents/agentic_rag.py`, after the agent definition:

```python
import src.agents.tools  # noqa: F401 — registers tools via @agentic_rag.tool decorators
```

This is the standard PydanticAI side-effect import pattern.

#### 2. Fix `search_graphiti` in `tools.py`

Replace:
```python
contexts, citations = await asyncio.to_thread(
    ctx.deps.graph_rag.generate, query, num_results=ctx.deps.top_k
)
```

With:
```python
contexts, citations = await ctx.deps.graph_rag.retrieve(query, top_k=ctx.deps.top_k)
```

`GraphRAG.retrieve` is already `async`; `asyncio.to_thread` is only for synchronous blocking calls.
The tool should call `retrieve()`, not `generate()` — generation is the agent's job.

---

### Code Quality / Design

#### 3. Remove global state mutation from `BasicRAG.__init__`

Remove:
```python
settings.qdrant_collection_name = qdrant_collection_name
```

Replace with an instance field:
```python
self.collection_name = qdrant_collection_name or settings.qdrant_collection_name
```

Pass `self.collection_name` explicitly in `vector_store.query(...)`.

#### 4. Fix `GraphRAG.initialize_graphiti_client`

Replace:
```python
self.graphiti = await GraphitiClient().create_client(...)
```

With:
```python
self.graphiti = await self.graphiti_client.create_client(...)
```

Avoids constructing a second unused `GraphitiClient` instance.

#### 5. Add `reset()` to `AgentDependencies`

Add a `reset()` method to `AgentDependencies` to encapsulate the per-turn state reset:

```python
def reset(self):
    self.citations = None
    self.contexts = None
```

Call `deps.reset()` before each turn instead of manually nulling two fields.

#### 6. Update system prompt

Revise `AGENTIC_RAG_INSTRUCTION` to:
- Describe both tools: `search_basic_rag` (vector database) and `search_graphiti` (knowledge graph).
- Provide guidance on when to use each (factual lookup → graph; document search → vector).

---

### Retrieval Improvements (BasicRAG)

#### 7. Score threshold filtering in `BasicRAG.retrieve()`

Add a `score_threshold` parameter (default from settings):
```python
def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[RetrievalInfo]:
```

After vector search, filter:
```python
retrieval_infos = [r for r in retrieval_infos if r.score >= score_threshold]
```

Add `retrieval_score_threshold: float = 0.0` to `ProjectSettings` in `src/settings.py`.

#### 8. Cross-encoder reranking in `BasicRAG.retrieve()`

If `self.cross_encoder` is not `None`, after threshold filtering:
- Call cross-encoder with `(query, doc.content)` pairs.
- Replace scores with cross-encoder scores.
- Re-sort descending by new score.

The `cross_encoder` field and the `CrossEncoder` dep class already exist in the codebase — just wire them.

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/agentic_rag.py` | Add `import src.agents.tools` at bottom |
| `src/agents/tools.py` | Fix `search_graphiti`: remove `asyncio.to_thread`, call `retrieve()` not `generate()` |
| `src/agents/deps.py` | Add `reset()` method |
| `src/retrieval/basic_rag.py` | Remove global mutation, add score threshold + cross-encoder reranking |
| `src/retrieval/graph_rag.py` | Fix `initialize_graphiti_client` to use `self.graphiti_client` |
| `src/prompts/agentic_rag_instruction.py` | Update to describe both tools |
| `src/settings.py` | Add `retrieval_score_threshold` field |

---

## Out of Scope

- Hybrid search fusion (BasicRAG + GraphRAG combined results)
- Protocol-based retrieval abstraction
- New files or modules
