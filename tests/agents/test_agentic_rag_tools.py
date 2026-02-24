import inspect
from src.agents.agentic_rag import agentic_rag
from src.agents import tools as tools_module
import src.agents.tools  # ensure import


def test_agentic_rag_has_tools_registered():
    tool_names = set(agentic_rag._function_toolset.tools.keys())
    assert "search_basic_rag" in tool_names
    assert "search_graphiti" in tool_names


def test_search_graphiti_calls_retrieve_not_generate():
    """Verify the tool implementation awaits retrieve(), not generate()."""
    source = inspect.getsource(tools_module.search_graphiti)
    assert "graph_rag.retrieve" in source
    assert "graph_rag.generate" not in source
    assert "asyncio.to_thread" not in source
