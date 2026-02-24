from typing import Optional
from pydantic import BaseModel


class AgentRequest(BaseModel):
    query: str
    collection_name: str = "research_papers"
    top_k: int = 5
    session_id: Optional[str] = None


class GraphAgentRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None
