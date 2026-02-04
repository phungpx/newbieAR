from pydantic import BaseModel


class RetrievalInfo(BaseModel):
    content: str
    source: str
    score: float
