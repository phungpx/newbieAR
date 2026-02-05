# minRAG

## Core Concepts

minrag = minRAG(
    qdrant_uri=
    qdrant_api_key=
    embedding....
)

- Synthesis

minRAG(

)

- Ingestion

minRAG.ingest_file(
    file_path=
    qdrant_collection_name=
)

minRAG.ingest_files(
    file_paths=[]
    qdrant_collection_name=
)

- Retrieval

minRAG.generate(
    query=
    top_k=
    qdrant_collection_name=
    return_context=
)

- Evaluation

minRAG.evaluate(
    
)


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
