from typing import Optional
from pydantic import BaseModel


class ChunkInfo(BaseModel):
    chunk_id: int
    text: str
    text_tokens: int
    contextualized_text: str
    contextualized_tokens: int
    filename: str
    mimetype: str

    # Hierarchical chunking metadata (optional)
    doc_items_refs: Optional[list[str]] = None
    doc_items_labels: Optional[list[str]] = None
