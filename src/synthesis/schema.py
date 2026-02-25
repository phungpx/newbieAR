from pydantic import BaseModel


class ContextScore(BaseModel):
    clarity: float
    depth: float
    structure: float
    relevance: float
