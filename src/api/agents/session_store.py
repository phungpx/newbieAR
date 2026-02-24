import uuid
from typing import Any, Optional


class InMemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, list[Any]] = {}

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, list[Any]]:
        if session_id and session_id in self._sessions:
            return session_id, list(self._sessions[session_id])
        new_id = str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id, []

    def save(self, session_id: str, messages: list[Any]):
        self._sessions[session_id] = list(messages)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
