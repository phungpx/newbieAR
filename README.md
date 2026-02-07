# newbieAR - Newbie Agentic RAG

## Core Concepts

## Ingestion Pipeline
- Orchestration: Apache Airflow for DAG-based scheduling and pipeline resilience.
- Storage: minIO (Local) & AWS S3 (Cloud) for raw and staged data artifacts.
- Knowledge Engines:
    - VectorDB: Qdrant & Milvus for high-dimensional semantic search.
    - GraphDB: Neo4j for relationship-mapping and complex entity retrieval.
- Feature Store: Feats for managing reusable ML features.
- Inference: vLLM hosting OpenAI-compatible endpoints for local embedding generation.

## Retrieval & Generation
- Hybrid Search: Combined semantic vector search and graph traversal.
- LLM Gateway: Support for any OpenAI-compatible third-party APIs (GPT-4, Claude, etc.).
- Observability: Langfuse for full-stack tracing, latency monitoring, and prompt versioning.

## Synthetic data generation Pipeline
- DeepEval (apply first -> enhance later)
- Ragas (apply first -> enhance later)

## Evaluation Metrics
- DeepEval
- Ragas

## UI: NotebookLM
- Ingestion
- Retrieval
- Citation Mechanism
