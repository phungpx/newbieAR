from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: int
    source: str
    content_snippet: str
    relevance_score: float
    cited_in_answer: bool = False


class RetrievalResult(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsageBreakdown(BaseModel):
    retrieval_embedding_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0


class TokenUsage(BaseModel):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    breakdown: TokenUsageBreakdown


class ToolCallInfo(BaseModel):
    tool: str
    query: str | None = None
    results_count: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    execution_time_ms: int = 0


class GraphPath(BaseModel):
    entities: list[str]
    relationship: str
    evidence_chunks: list[int] = Field(default_factory=list)


class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    created_at: datetime
    message: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: dict[str, Any] | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    citations: list[Citation]
    retrieval_time_ms: int
    num_results: int


class GraphRAGResponse(RetrievalResponse):
    graph_paths: list[GraphPath] = Field(default_factory=list)


class GenerateResponse(RetrievalResponse):
    generated_answer: str


class AgentChatResponse(BaseModel):
    message: str
    tool_calls: list[ToolCallInfo]
    citations: list[Citation]
    session_id: str
    user_id: str
    token_usage: TokenUsage
    response_time_ms: int


class ErrorResponse(BaseModel):
    error: dict[str, Any]
