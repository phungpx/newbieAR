from src.agents.agentic_rag import agentic_rag


def test_agentic_rag_has_tools_registered():
    tool_names = set(agentic_rag._function_toolset.tools.keys())
    assert "search_basic_rag" in tool_names
    assert "search_graphiti" in tool_names
