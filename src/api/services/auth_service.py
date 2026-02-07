import hashlib
import secrets
from datetime import datetime
from src.api.models import APIKey, APIKeyCreate
from src.settings import settings


def _hash_key(plaintext: str) -> str:
    """Hash an API key using SHA-256 (appropriate for random API tokens)"""
    return hashlib.sha256(plaintext.encode()).hexdigest()


class AuthService:
    """In-memory API key storage (Redis-ready structure)"""

    def __init__(self):
        self.keys: dict[str, APIKey] = {}
        self._init_admin_key()

    def _init_admin_key(self):
        """Initialize admin API key for development"""
        admin_key = settings.api.admin_api_key
        hashed = _hash_key(admin_key)
        self.keys[hashed] = APIKey(
            api_key=hashed,
            user_id="admin",
            name="Admin Key",
            permissions=["ingest", "retrieval", "agents", "admin"],
            rate_limit_tier="premium",
            is_active=True,
        )

    def generate_api_key(self) -> str:
        """Generate new API key with newbie_ prefix"""
        return f"newbie_{secrets.token_hex(16)}"

    def create_key(self, key_create: APIKeyCreate) -> tuple[str, APIKey]:
        """Create new API key and return plaintext + stored model"""
        plaintext_key = self.generate_api_key()
        hashed_key = _hash_key(plaintext_key)

        api_key = APIKey(
            api_key=hashed_key,
            user_id=key_create.user_id,
            name=key_create.name,
            permissions=key_create.permissions,
            rate_limit_tier=key_create.rate_limit_tier,
            expires_at=key_create.expires_at,
        )

        self.keys[hashed_key] = api_key
        return plaintext_key, api_key

    def validate_key(self, plaintext_key: str) -> APIKey | None:
        """Validate API key and return associated data"""
        hashed = _hash_key(plaintext_key)
        api_key = self.keys.get(hashed)

        if not api_key:
            return None
        if not api_key.is_active:
            return None
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        return api_key

    def has_permission(self, api_key: APIKey, permission: str) -> bool:
        """Check if API key has specific permission"""
        return permission in api_key.permissions or "admin" in api_key.permissions


# Global auth service instance
auth_service = AuthService()
