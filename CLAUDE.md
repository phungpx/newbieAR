# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**newbieAR** (Newbie Agentic RAG) is an end-to-end RAG evaluation platform. It synthesizes golden test datasets from documents, runs retrieval and generation, and evaluates the pipeline using deepeval metrics logged to the Confident AI platform.

## Package Management

Uses `uv` exclusively. Never use `pip` or `poetry`.

```bash
uv sync          # install dependencies
uv run python    # run python with project env
```

## Common Commands

```bash
# Ingest a document into Qdrant
uv run python -m src.ingestion.ingest_vectordb \
  --file_path ./data/papers/files/docling.pdf \
  --collection_name research_papers

# Interactive RAG CLI
uv run python -m src.retrieval.basic_rag \
  --qdrant_collection_name research_papers --top_k 10

# Synthesize golden test cases from docs
uv run python -m src.synthesis.synthesize \
  --topic paper --file_dir data/papers/files --output_dir data/goldens

# Run evaluation over golden JSON files
uv run python -m src.evaluation.evaluate \
  --file_dir data/goldens \
  --retrieval_window_size 5 \
  --collection_name research_papers
```

## Architecture

### Core Pipeline

```
Documents → Ingestion → Qdrant VectorDB
                              ↓
                         BasicRAG (retrieve + generate via OpenAI-compatible LLM)
                              ↓
             Golden dataset (Synthesizer) → JSON files in data/goldens/
                              ↓
                     Evaluation (deepeval metrics + Bedrock critic)
                              → Confident AI dashboard
```

### Source Layout

| Path | Responsibility |
|------|---------------|
| `src/settings.py` | `ProjectSettings` singleton — all config via `.env`, never instantiate settings classes directly |
| `src/deps/` | Injectable clients: `OpenAIEmbedding`, `OpenAILLMClient`, `QdrantVectorStore`, `DocumentLoader`, `DocumentChunker`, etc. |
| `src/models/` | Pydantic data models: `ChunkInfo`, `ChunkStrategy`, `RetrievalInfo`, `Payload` |
| `src/ingestion/` | `VectorDBIngestion` — convert PDF → markdown, chunk, embed, store in Qdrant |
| `src/retrieval/` | `BasicRAG` (vector search + generation), `GraphRAG` (Graphiti/Neo4j) |
| `src/synthesis/` | `Synthesizer` pipeline using deepeval — generates golden (input, expected_output, context) triples |
| `src/evaluation/` | deepeval metrics (AnswerRelevancy, Faithfulness, ContextualPrecision/Recall/Relevancy) measured with Bedrock critic |
| `src/agents/` | Agentic RAG variants (`agentic_basic_rag`, `agentic_graph_rag`) |
| `src/prompts/` | Generation and agentic RAG instruction prompts |
| `infras/` | Docker Compose files for Qdrant, Neo4j, MinIO, Airflow; Airflow DAGs for scheduled ingestion |

### Key Design Patterns

- **Settings**: `settings = ProjectSettings()` at module level in `src/settings.py`. All other modules import this singleton. Settings groups (`openai_llm`, `qdrant_vector_store`, etc.) are accessed as properties.
- **Chunking strategies**: `ChunkStrategy.HYBRID` (default) or `ChunkStrategy.HIERARCHICAL`. Passed through `VectorDBIngestion` and `DocumentChunker`.
- **Critique model**: AWS Bedrock (`BedrockLLMWrapper` / `AmazonBedrockModel`) is used for both evaluation and synthesis. Requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` env vars.
- **Golden format**: Each golden is saved as a standalone JSON with `input`, `expectedOutput`, `context`, `sourceFile`, `additionalMetadata` fields. Evaluation appends `actual_output`, `retrieval_contexts`, and `metrics` in-place.
- **deepeval login**: `deepeval.login(settings.confident_api_key)` is called at import time in `evaluate.py`; metrics are logged to Confident AI with `_log_metric_to_confident=True`.

### Infrastructure (Docker Compose)

| Service | File |
|---------|------|
| Qdrant (vector DB) | `infras/docker-compose.qdrant.yaml` |
| Neo4j (graph DB) | `infras/docker-compose.neo4j.yaml` |
| MinIO (object storage) | `infras/docker-compose.minio.yaml` |
| Airflow (orchestration) | `infras/docker-compose.airflow.yaml` |

### Data Directories

- `data/papers/files/` — raw PDF inputs
- `data/papers/docs/` — converted markdown documents
- `data/papers/chunks/` — chunked JSON files
- `data/wikipedia/files/` — Wikipedia markdown articles
- `data/goldens/` — synthesized golden test cases (JSON)
- `logs/` — evaluation logs
