from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CitationInfo(BaseModel):
    """Information about a single citation."""

    citation_number: int  # [1], [2], etc.
    content: str
    source: str
    score: float
    page_number: Optional[int] = None  # For future PDF page tracking


class ToolCallInfo(BaseModel):
    """Information about a tool call execution."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result: Optional[Any] = None  # Tool result (may be large, optional for display)
    execution_time: Optional[float] = None  # Execution time in seconds
    status: str = "success"  # "success", "error", "pending"


class TurnMetadata(BaseModel):
    """Metadata for a single conversation turn."""

    turn_id: str
    tool_calls: List[ToolCallInfo] = Field(default_factory=list)
    citations: List[CitationInfo] = Field(default_factory=list)
    rag_mode: str  # "basic", "agentic", "graph"
    collection: str
    timestamp: datetime = Field(default_factory=datetime.now)


class CitedResponse(BaseModel):
    """Response with citations and tool calls."""

    answer: str  # Original answer text
    citations: List[CitationInfo] = Field(
        default_factory=list
    )  # Ordered by citation number
    tool_calls: List[ToolCallInfo] = Field(default_factory=list)  # Tool calls executed
    turn_metadata: Optional[TurnMetadata] = None  # Full turn metadata
    metadata: dict = Field(default_factory=dict)  # Store RAG mode, collection, etc.
