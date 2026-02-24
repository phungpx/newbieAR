# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**newbieAR** — Newbie Agentic RAG: an end-to-end pipeline for building, evaluating, and improving a Retrieval-Augmented Generation system. The pipeline covers document ingestion → retrieval (vector + graph) → agentic RAG → synthetic test data generation → evaluation.

## Package Management & Setup

Uses `uv` exclusively. Never use `pip` or `poetry`.

```bash
uv sync                    # install all dependencies
uv sync --extra test       # install with test dependencies
```

## Common Commands

### Run Tests
```bash
uv run pytest tests/                                     # all tests
uv run pytest tests/retrieval/test_basic_rag.py          # single file
uv run pytest tests/agents/test_agentic_rag_tools.py -v  # single file, verbose
```

`asyncio_mode = "auto"` is configured in `pyproject.toml`, so async test functions work without decorators.

### Infrastructure (Docker)
```bash
docker compose -f infras/docker-compose.qdrant.yaml up -d   # Qdrant vector DB (ports 6333/6334)
docker compose -f infras/docker-compose.neo4j.yaml up -d    # Neo4j for Graphiti
docker compose -f infras/docker-compose.minio.yaml up -d    # MinIO object storage
```

### CLI Entry Points (all support `__main__`)
```bash
# Ingest into vector DB
uv run python -m src.ingestion.ingest_vectordb \
  --file_path data/papers/files/docling.pdf \
  --collection_name research_papers \
  --chunk_strategy hybrid

# Ingest into graph DB (Neo4j via Graphiti)
uv run python -m src.ingestion.ingest_graphdb \
  --file_path data/papers/files/docling.pdf

# Interactive BasicRAG CLI
uv run python -m src.retrieval.basic_rag \
  --qdrant_collection_name research_papers --top_k 10

# Interactive GraphRAG CLI
uv run python -m src.retrieval.graph_rag

# Agentic RAG (pydantic-ai streaming agent)
uv run python -m src.agents.agentic_rag \
  --collection_name research_papers --top_k 5

# Synthesize golden test cases from documents
uv run python -m src.synthesis.synthesize \
  --topic paper --file_dir data/papers/files --output_dir data/goldens

# Evaluate with deepeval metrics
uv run python -m src.evaluation.evaluate \
  --file_dir data/goldens --retrieval_window_size 5
```

## Architecture

### Settings (`src/settings.py`)
A `ProjectSettings` singleton is instantiated at module import time and sets Langfuse env vars as a side effect. It uses pydantic-settings reading from `.env`. Access grouped sub-settings via properties: `settings.openai_llm`, `settings.qdrant_vector_store`, `settings.langfuse`, etc.

### Infrastructure Clients (`src/deps/`)
Thin wrappers around external services, re-exported from `src/deps/__init__.py`:
- `OpenAIEmbedding` — text embedding (OpenAI-compatible endpoint)
- `OpenAILLMClient` — chat completions (OpenAI-compatible endpoint)
- `QdrantVectorStore` — Qdrant vector DB operations
- `GraphitiClient` — Neo4j graph DB via graphiti-core
- `DocumentLoader` — PDF/file conversion via docling
- `DocumentChunker` — text chunking; strategies: `hybrid` (default for vector DB) or `hierarchical` (default for graph DB)
- `CrossEncoder` client (optional, set manually on `BasicRAG.cross_encoder` to enable reranking)

### Retrieval (`src/retrieval/`)
- **`BasicRAG`** — async `retrieve()` + async `generate()`. Embeds query → queries Qdrant → optional score threshold filtering → optional cross-encoder reranking. `cross_encoder` is `None` by default; set it externally to enable reranking.
- **`GraphRAG`** — async retrieval from Neo4j via graphiti-core. Requires async initialization (`initialize_graphiti_client()`). Uses hybrid search: BM25 + cosine similarity + BFS, reranked with RRF.

### Agentic RAG (`src/agents/`)
Built on **pydantic-ai** (not LangChain/LangGraph). The `agentic_rag` agent has two tools:
- `search_basic_rag` — calls `BasicRAG.retrieve()`
- `search_graphiti` — calls `GraphRAG.retrieve()`

`AgentDependencies` is a dataclass holding `BasicRAG`, `GraphRAG`, `top_k`, and mutable `contexts`/`citations` fields. Call `deps.clear_context()` between turns to reset retrieved context.

### Ingestion (`src/ingestion/`)
- **`VectorDBIngestion`**: `DocumentLoader` → `DocumentChunker` → embed → upsert to Qdrant. Creates the Qdrant collection in `__init__`.
- **`GraphitiIngestion`**: `DocumentChunker` → add episodes to Neo4j via `graphiti.add_episode()`.

### Synthesis (`src/synthesis/`)
Uses deepeval's `Synthesizer` with AWS Bedrock as the LLM (not the OpenAI endpoint). Generates golden test cases (input/expected_output/context) from raw documents using configurable `FiltrationConfig`, `EvolutionConfig`, `StylingConfig`, and `ContextConstructionConfig`. Goldens are saved as individual JSON files via `save_goldens_to_files()` in `src/synthesis/utils.py`.

### Evaluation (`src/evaluation/`)
Uses deepeval metrics: `AnswerRelevancy`, `Faithfulness`, `ContextualPrecision`, `ContextualRecall`, `ContextualRelevancy`. The critique model is AWS Bedrock wrapped in `BedrockLLMWrapper`. Results (score, reason, verdicts, token usage) are written back into the golden JSON files. Requires `CONFIDENT_API_KEY` to log to Confident AI.

### Data Models (`src/models/`)
Key types: `ChunkInfo`, `ChunkStrategy` (enum: `hybrid`/`hierarchical`), `RetrievalInfo` (content, source, score), `GraphitiEdgeInfo`/`GraphitiNodeInfo`/`GraphitiEpisodeInfo`.

## Key `.env` Variables

```
# LLM
LLM_MODEL=...
LLM_API_KEY=...
LLM_BASE_URL=...

# Embedding
EMBEDDING_BASE_URL=...
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=...
EMBEDDING_DIMENSIONS=...

# Qdrant
QDRANT_URI=...
QDRANT_COLLECTION_NAME=...

# Neo4j (for Graphiti)
GRAPH_DB_URI=...
GRAPH_DB_USERNAME=...
GRAPH_DB_PASSWORD=...

# AWS Bedrock (synthesis + evaluation critique model)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
CRITIQUE_MODEL_NAME=...
CRITIQUE_MODEL_REGION_NAME=...

# deepeval / Confident AI
CONFIDENT_API_KEY=...

# Langfuse (observability)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...
```

## Important Constraints

- `VectorDBIngestion.__init__` mutates `settings.qdrant_collection_name` as a side effect — avoid this pattern in new code. `BasicRAG` stores `collection_name` locally and does **not** mutate global settings.
- All retrieval methods are `async`; use `asyncio.to_thread()` for sync calls inside async methods.
- The `src/settings.py` module sets `os.environ` keys at import time — importing it triggers side effects.
