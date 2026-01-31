from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DocStatus(str, Enum):
    PENDING = "pending"
    CONVERTING = "converting"
    CHUNKING = "chunking"
    SUCCESS = "success"
    FAILED = "failed"


class DocInfo(BaseModel):
    file_name: str
    format: str
    status: str
    markdown_length: int = 0
    markdown_preview: str = ""
    markdown_path: Optional[str] = None
    chunk_path: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None

    class Config:
        use_enum_values = True


class ChunkMetadata(BaseModel):
    chunk_id: int
    text: str
    text_tokens: int
    contextualized_text: str
    contextualized_tokens: int
    filename: str
    mimetype: str
