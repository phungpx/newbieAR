from datetime import datetime
from pydantic import BaseModel, Field


class APIKey(BaseModel):
    api_key: str
    user_id: str
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["ingest", "retrieval", "agents"])
    rate_limit_tier: str = "standard"  # or "premium"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None


class APIKeyCreate(BaseModel):
    user_id: str
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["ingest", "retrieval", "agents"])
    rate_limit_tier: str = "standard"
    expires_at: datetime | None = None
