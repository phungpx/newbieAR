import pytest
from unittest.mock import MagicMock
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel

from src.api.services.agent_runner import parse_model_id, make_llm_model, SUPPORTED_AGENTS


def test_parse_model_id_agent_only():
    agent_type, llm = parse_model_id("basic-rag")
    assert agent_type == "basic-rag"
    assert llm is None


def test_parse_model_id_with_llm():
    agent_type, llm = parse_model_id("graph-rag/gemini-2.5-flash")
    assert agent_type == "graph-rag"
    assert llm == "gemini-2.5-flash"


def test_parse_model_id_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        parse_model_id("unknown-agent")


def test_parse_model_id_unknown_agent_with_slash_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        parse_model_id("bad-agent/gpt-4o")


def test_make_llm_model_returns_openai_for_default():
    model = make_llm_model(None)
    assert isinstance(model, OpenAIChatModel)


def test_make_llm_model_returns_openai_for_non_gemini():
    model = make_llm_model("gpt-4o")
    assert isinstance(model, OpenAIChatModel)


def test_make_llm_model_returns_google_for_gemini():
    model = make_llm_model("gemini-2.5-flash")
    assert isinstance(model, GoogleModel)


def test_supported_agents_contains_both():
    assert "basic-rag" in SUPPORTED_AGENTS
    assert "graph-rag" in SUPPORTED_AGENTS
