from src.agents.deps import AgentDependencies


def test_clear_context_clears_citations_and_contexts():
    deps = AgentDependencies(
        citations=["source_a"],
        contexts=["some context"],
    )
    deps.clear_context()
    assert deps.citations is None
    assert deps.contexts is None


def test_clear_context_is_idempotent():
    deps = AgentDependencies()
    deps.clear_context()  # should not raise
    assert deps.citations is None
    assert deps.contexts is None
