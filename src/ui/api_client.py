import os
import httpx

BASE_URL: str = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
API_PREFIX: str = "/api/v1"

client: httpx.Client = httpx.Client(base_url=BASE_URL)
stream_client: httpx.Client = httpx.Client(base_url=BASE_URL, timeout=None)


def api_url(path: str) -> str:
    """Build a full API URL from a path relative to the API prefix."""
    return f"{BASE_URL}{API_PREFIX}{path}"
