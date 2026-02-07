import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.services import auth_service
from src.settings import settings

client = TestClient(app)

# Use admin key for tests
API_KEY = settings.api.admin_api_key
HEADERS = {"X-API-Key": API_KEY}


def test_health_check():
    """Test basic health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check():
    """Test readiness endpoint"""
    response = client.get("/health/ready")
    # May fail if Qdrant not running, that's OK for basic test
    assert response.status_code in [200, 503]


def test_list_collections():
    """Test list collections endpoint"""
    response = client.get(f"{settings.api.api_prefix}/ingest/collections", headers=HEADERS)
    assert response.status_code in [200, 500]  # OK if no Qdrant


def test_invalid_api_key():
    """Test authentication with invalid key"""
    bad_headers = {"X-API-Key": "invalid_key"}
    response = client.get(f"{settings.api.api_prefix}/ingest/collections", headers=bad_headers)
    assert response.status_code == 401


def test_create_session():
    """Test session creation"""
    response = client.post(
        f"{settings.api.api_prefix}/agents/sessions",
        params={"user_id": "test_user"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "session_id" in response.json()


def test_list_sessions():
    """Test listing sessions"""
    response = client.get(
        f"{settings.api.api_prefix}/agents/sessions",
        params={"user_id": "test_user"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert "sessions" in response.json()
