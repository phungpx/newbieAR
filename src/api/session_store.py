import uuid
from dataclasses import dataclass, field
from pydantic_ai.messages import ModelMessage
from src.agents.deps import AgentDependencies


@dataclass
class SessionState:
    deps: AgentDependencies
    messages: list[ModelMessage] = field(default_factory=list)
    collection_name: str = ""
    top_k: int = 5


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def create(self, deps: AgentDependencies, collection_name: str, top_k: int) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionState(
            deps=deps,
            collection_name=collection_name,
            top_k=top_k,
        )
        return session_id

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
