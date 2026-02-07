import uuid
from datetime import datetime, timedelta
from typing import Any
from pydantic import BaseModel
from src.settings import settings


class Session(BaseModel):
    session_id: str
    user_id: str
    messages: list[dict[str, Any]]  # Simplified message storage
    created_at: datetime
    last_accessed_at: datetime


class SessionManager:
    """In-memory session storage (Redis-ready structure)"""

    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def create_session(self, user_id: str) -> str:
        """Create new session and return session_id"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = Session(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
        )
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str, user_id: str) -> Session | None:
        """Get session by ID, validate user ownership"""
        session = self.sessions.get(session_id)
        if session and session.user_id == user_id:
            # Update last accessed
            session.last_accessed_at = datetime.utcnow()
            return session
        return None

    def add_message(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None):
        """Add message to session history"""
        session = self.sessions.get(session_id)
        if not session:
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(metadata or {}),
        }

        session.messages.append(message)

        # Trim to max history size
        max_messages = settings.sessions.max_history_messages
        if len(session.messages) > max_messages:
            session.messages = session.messages[-max_messages:]

    def get_history(self, session_id: str, user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get session message history with pagination"""
        session = self.get_session(session_id, user_id)
        if not session:
            return []

        total = len(session.messages)
        start = max(0, total - offset - limit)
        end = total - offset if offset > 0 else total

        return session.messages[start:end]

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete session if user owns it"""
        session = self.sessions.get(session_id)
        if session and session.user_id == user_id:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self, user_id: str) -> list[Session]:
        """List all sessions for a user"""
        return [s for s in self.sessions.values() if s.user_id == user_id]

    def cleanup_expired(self):
        """Remove expired sessions (TTL-based)"""
        ttl = timedelta(hours=settings.sessions.session_ttl_hours)
        now = datetime.utcnow()

        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_accessed_at > ttl
        ]

        for sid in expired:
            del self.sessions[sid]


# Global session manager instance
session_manager = SessionManager()
