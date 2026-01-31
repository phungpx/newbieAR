from pydantic import BaseModel


class ChunkInfo(BaseModel):
    chunk_id: int
    text: str
    text_tokens: int
    contextualized_text: str
    contextualized_tokens: int
    filename: str
    mimetype: str
