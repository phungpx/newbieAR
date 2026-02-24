from src.agents.deps import AgentDependencies


def test_reset_clears_citations_and_contexts():
    deps = AgentDependencies(
        citations=["source_a"],
        contexts=["some context"],
    )
    deps.reset()
    assert deps.citations is None
    assert deps.contexts is None


def test_reset_is_idempotent():
    deps = AgentDependencies()
    deps.reset()  # should not raise
    assert deps.citations is None
    assert deps.contexts is None
