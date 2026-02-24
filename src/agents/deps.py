from dataclasses import dataclass
from src.retrieval.graph_rag import GraphRAG
from src.retrieval.basic_rag import BasicRAG


@dataclass
class AgentDependencies:
    basic_rag: BasicRAG | None = None
    graph_rag: GraphRAG | None = None
    top_k: int = 5
    citations: list[str] | None = None
    contexts: list[str] | None = None

    def reset(self):
        self.citations = None
        self.contexts = None
