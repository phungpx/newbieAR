from __future__ import annotations

import hashlib

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from redis.asyncio import Redis

from src.api.schemas import ChatMessage

_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(list[ModelMessage])

SESSION_PREFIX = "session:"
DEFAULT_TTL_SECONDS = 86400  # 24 h


def derive_session_key(model: str, messages: list[ChatMessage]) -> str:
    """Stable session key: SHA-256 of (model + first user message content).

    Open WebUI always sends the full message history, so we anchor the key to
    the very first user message which never changes across turns.
    """
    first_user_content = next(
        (m.content for m in messages if m.role == "user"), ""
    )
    raw = f"{model}:{first_user_content}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{SESSION_PREFIX}{digest}"


async def load_messages(redis: Redis, key: str) -> list[ModelMessage]:
    raw = await redis.get(key)
    if not raw:
        return []
    return _adapter.validate_json(raw)


async def save_messages(
    redis: Redis,
    key: str,
    messages: list[ModelMessage],
    ttl: int = DEFAULT_TTL_SECONDS,
) -> None:
    await redis.set(key, _adapter.dump_json(messages), ex=ttl)
