import json
from src.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionChunk,
    DeltaContent,
    ChunkChoice,
    ModelsResponse,
    ModelInfo,
)


def test_chat_completion_request_defaults_stream_true():
    req = ChatCompletionRequest(
        model="basic-rag",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert req.stream is True


def test_chat_completion_request_parses_messages():
    data = {
        "model": "basic-rag",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
    }
    req = ChatCompletionRequest(**data)
    assert len(req.messages) == 2
    assert req.messages[0].role == "user"


def test_chunk_serializes_to_openai_format():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc",
        created=1234567890,
        model="basic-rag",
        choices=[ChunkChoice(delta=DeltaContent(content="Hello"))],
    )
    d = chunk.model_dump()
    assert d["object"] == "chat.completion.chunk"
    assert d["choices"][0]["delta"]["content"] == "Hello"
    assert d["choices"][0]["finish_reason"] is None


def test_models_response_structure():
    resp = ModelsResponse(data=[ModelInfo(id="basic-rag"), ModelInfo(id="graph-rag")])
    assert resp.object == "list"
    assert len(resp.data) == 2
