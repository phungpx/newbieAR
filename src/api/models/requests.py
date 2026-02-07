from typing import Any
from pydantic import BaseModel, Field
from src.models import ChunkStrategy


class IngestRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=100)
    chunk_strategy: str = Field(default=ChunkStrategy.HYBRID.value)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=1000)
    collection_name: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    return_embeddings: bool = False


class GraphRAGRequest(RetrievalRequest):
    graph_depth: int = Field(default=2, ge=1, le=5)
    enable_reranking: bool = True


class GenerateRequest(RetrievalRequest):
    pass


class AgentChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=2000)
    collection_name: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_history: bool = True
