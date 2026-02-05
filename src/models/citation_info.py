from typing import Optional, List
from pydantic import BaseModel


class CitationInfo(BaseModel):
    """Information about a single citation."""
    citation_number: int  # [1], [2], etc.
    content: str
    source: str
    score: float
    page_number: Optional[int] = None  # For future PDF page tracking


class CitedResponse(BaseModel):
    """Response with citations."""
    answer: str  # Original answer text
    citations: List[CitationInfo]  # Ordered by citation number
    metadata: dict = {}  # Store RAG mode, collection, etc.
