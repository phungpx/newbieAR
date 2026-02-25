# Synthesis Pipeline

The `src/synthesis/` module builds high-quality RAG evaluation datasets from raw documents. It produces **goldens** — (context, question, expected answer) triples — that feed directly into DeepEval's evaluation framework.

## Overview

```
Document (PDF/DOCX/…)
        │
        ▼
  DocumentChunker          ← hierarchical chunking via Docling
        │
        ▼
  OpenAIEmbedding          ← embed all chunks
        │
        ▼
  QdrantVectorStore         ← temporary per-document collection
        │
        ▼
  generate_contexts()       ← quality-filtered seed selection
  ┌─────────────────────────────────────────────────────────┐
  │  for each context:                                       │
  │    repeat up to max_tries:                              │
  │      pick random candidate chunk                        │
  │      evaluate_chunk() → average score of               │
  │        clarity + depth + structure + relevance          │
  │      if score ≥ quality_threshold → accept as seed      │
  │    query Qdrant for (context_size - 1) nearest chunks   │
  │    context = [seed] + neighbors                         │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  Synthesizer               ← DeepEval: generates Q&A pairs
        │
        ▼
  save_goldens_to_files()   ← JSON files under output_dir/
```

## Module Structure

```
src/synthesis/
├── synthesize.py           # Entry point — orchestrates the full pipeline
├── generate_contexts.py    # Context building + chunk quality evaluation
├── schema.py               # Pydantic schema for LLM-scored chunk quality
└── prompts/
    └── context_evaluation.py   # Prompt template for chunk scoring
```

---

## Components

### `synthesize.py` — Pipeline Entry Point

Configures and runs the full synthesis pipeline. Run directly as a script:

```bash
python -m src.synthesis.synthesize --topic paper --file_dir data/papers/ --output_dir data/goldens/
```

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--topic` | `article` | `paper` (research papers) or `article` (Wikipedia) |
| `--file_dir` | `data/wikipedia/files` | Directory of source documents |
| `--output_dir` | `data/goldens` | Destination for generated golden JSON files |

**Internals:**

- Instantiates a `GPTModel` (pointing at the configured LLM endpoint) as both the synthesis model and the chunk quality critic.
- Configures a `FiltrationConfig` (filters low-quality generated questions), `EvolutionConfig` (7 evolution strategies applied in 2 rounds), and a `StylingConfig` (topic-specific I/O format instructions).
- Calls `generate_contexts()` per document, then `synthesizer.generate_goldens_from_contexts()` to produce Q&A pairs, and finally `save_goldens_to_files()` to persist results.

---

### `generate_contexts.py` — Core Functions

#### `generate_contexts` (async)

```python
async def generate_contexts(
    file_path: str,
    model: DeepEvalBaseLLM,
    embedder: OpenAIEmbedding,
    vector_store: QdrantVectorStore,
    embedding_size: int,
    num_contexts: int = 5,
    context_size: int = 3,
    quality_threshold: float = 0.6,
    max_tries: int = 10,
) -> list[list[str]]
```

Builds `num_contexts` semantically coherent context windows from a single document.

**How it works:**
1. Chunks the document hierarchically (via Docling).
2. Embeds all chunks and stores them in a temporary Qdrant collection.
3. For each context slot, draws random candidate chunks up to `max_tries` times, scoring each with `evaluate_chunk`. The first candidate scoring `≥ quality_threshold` becomes the seed.
4. Retrieves the `context_size - 1` most similar chunks to the seed from Qdrant.
5. Returns `[seed] + neighbors` as one context window.
6. Cleans up the Qdrant collection in a `finally` block.

If no candidate passes the threshold within `max_tries` attempts, that context slot is skipped with a warning — the returned list may be shorter than `num_contexts`.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `file_path` | — | Path to the source document |
| `model` | — | LLM instance used for chunk quality scoring |
| `embedder` | — | Embedding model for vectorising chunks |
| `vector_store` | — | Qdrant instance for similarity search |
| `embedding_size` | — | Vector dimensionality (must match `embedder`) |
| `num_contexts` | `5` | Target number of context windows |
| `context_size` | `3` | Total chunks per context (seed + neighbors) |
| `quality_threshold` | `0.6` | Minimum average quality score `[0, 1]` for a seed |
| `max_tries` | `10` | Max candidate draws before skipping a context slot |

---

#### `evaluate_chunk` (async)

```python
async def evaluate_chunk(model: DeepEvalBaseLLM, chunk: str) -> float
```

Scores a single chunk on four dimensions via an LLM call, returning their mean:

| Dimension | What it measures |
|---|---|
| `clarity` | How clear and comprehensible the text is |
| `depth` | Level of detail and original insight |
| `structure` | Logical organisation and flow |
| `relevance` | Focus on the main topic without digressions |

The prompt is defined in `prompts/context_evaluation.py`. The LLM response is parsed into a `ContextScore` Pydantic model and the four floats are averaged.

---

#### `save_goldens_to_files`

```python
def save_goldens_to_files(goldens: list[Golden], output_dir: str = "goldens")
```

Persists each `Golden` as a JSON file under `output_dir/<source_file_stem>/<uuid>.json`, creating subdirectories as needed.

---

### `schema.py` — `ContextScore`

```python
class ContextScore(BaseModel):
    clarity: float
    depth: float
    structure: float
    relevance: float
```

Pydantic model used as the structured output schema when calling `evaluate_chunk`. Each field is a float in `[0, 1]`.

---

### `prompts/context_evaluation.py` — Chunk Scoring Prompt

Contains the `CONTEXT_EVALUATION` string — a few-shot prompt that instructs the LLM to return a JSON object with `clarity`, `depth`, `structure`, and `relevance` scores for a given `{context}` string. Used exclusively by `evaluate_chunk`.

---

## Data Flow Example

```
data/papers/attention_is_all_you_need.pdf
    → 42 chunks extracted
    → 42 vectors stored in qdrant:synthesis_attention_is_all_you_need
    → 5 context windows built (each 3 chunks)
        chunk #7  score=0.81 ✓  → seed + 2 neighbors  → context_0
        chunk #23 score=0.73 ✓  → seed + 2 neighbors  → context_1
        ...
    → Synthesizer generates 5 goldens (1 Q&A per context)
    → data/goldens/attention_is_all_you_need/<uuid>.json  × 5
```

## Configuration

All service endpoints (LLM, embedder, Qdrant) are read from environment variables via `src/settings.py` (`ProjectSettings`). See `.env.example` for the full list.

Key settings consumed by this module:

| Setting | Used for |
|---|---|
| `llm_model` / `llm_api_key` / `llm_base_url` | `GPTModel` (synthesis + critic) |
| `embedding_model` / `embedding_api_key` / `embedding_base_url` | `OpenAIEmbedding` |
| `embedding_dimensions` | `embedding_size` passed to `generate_contexts` |
| `qdrant_uri` / `qdrant_api_key` | `QdrantVectorStore` |
