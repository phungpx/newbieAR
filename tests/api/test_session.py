import pytest
import fakeredis.aioredis as fakeredis
from pydantic_ai.messages import ModelRequest, UserPromptPart

from src.api.services.session import derive_session_key, load_messages, save_messages
from src.api.schemas import ChatMessage


@pytest.mark.asyncio
async def test_load_returns_empty_list_for_new_session():
    redis = fakeredis.FakeRedis()
    messages = await load_messages(redis, "session:nonexistent")
    assert messages == []


@pytest.mark.asyncio
async def test_save_and_load_roundtrip():
    redis = fakeredis.FakeRedis()
    original = [ModelRequest(parts=[UserPromptPart(content="test")])]
    await save_messages(redis, "session:test", original, ttl=60)
    loaded = await load_messages(redis, "session:test")
    assert len(loaded) == 1
    assert loaded[0].parts[0].content == "test"


def test_derive_session_key_stable_across_turns():
    msgs_turn1 = [ChatMessage(role="user", content="What is RAG?")]
    msgs_turn2 = [
        ChatMessage(role="user", content="What is RAG?"),
        ChatMessage(role="assistant", content="RAG is..."),
        ChatMessage(role="user", content="Tell me more"),
    ]
    key1 = derive_session_key("basic-rag", msgs_turn1)
    key2 = derive_session_key("basic-rag", msgs_turn2)
    assert key1 == key2  # same first user message → same session


def test_derive_session_key_differs_by_model():
    msgs = [ChatMessage(role="user", content="Hello")]
    k1 = derive_session_key("basic-rag", msgs)
    k2 = derive_session_key("graph-rag", msgs)
    assert k1 != k2


def test_derive_session_key_has_prefix():
    msgs = [ChatMessage(role="user", content="Hello")]
    key = derive_session_key("basic-rag", msgs)
    assert key.startswith("session:")
