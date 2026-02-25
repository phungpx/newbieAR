import pytest
from pydantic import ValidationError
from src.api.schemas import CreateSessionRequest, CreateSessionResponse, DeleteSessionResponse, ChatRequest, CompletionResponse


def test_create_session_request_valid():
    req = CreateSessionRequest(collection_name="research_papers", top_k=5)
    assert req.collection_name == "research_papers"
    assert req.top_k == 5


def test_create_session_request_default_top_k():
    req = CreateSessionRequest(collection_name="test")
    assert req.top_k == 5


def test_create_session_request_missing_collection_name():
    with pytest.raises(ValidationError):
        CreateSessionRequest()


def test_create_session_response():
    resp = CreateSessionResponse(session_id="abc123", collection_name="test", top_k=5)
    assert resp.session_id == "abc123"


def test_chat_request_valid():
    req = ChatRequest(session_id="abc123", message="What is docling?")
    assert req.message == "What is docling?"


def test_chat_request_message_too_long():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="x" * 1001)


def test_chat_request_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="")


def test_chat_request_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc123", message="   ")


async def test_app_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_completion_response():
    resp = CompletionResponse(
        text="Docling is a document conversion library.",
        contexts=["ctx1", "ctx2"],
        citations=["cite1"],
    )
    assert resp.text == "Docling is a document conversion library."
    assert resp.contexts == ["ctx1", "ctx2"]
    assert resp.citations == ["cite1"]


def test_completion_response_empty_lists():
    resp = CompletionResponse(text="answer", contexts=[], citations=[])
    assert resp.contexts == []
    assert resp.citations == []
