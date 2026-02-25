from pydantic import BaseModel, field_validator


class CreateSessionRequest(BaseModel):
    collection_name: str
    top_k: int = 5


class CreateSessionResponse(BaseModel):
    session_id: str
    collection_name: str
    top_k: int


class DeleteSessionResponse(BaseModel):
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message must be 1000 characters or fewer")
        return v


class CompletionResponse(BaseModel):
    text: str
    contexts: list[str]
    citations: list[str]
