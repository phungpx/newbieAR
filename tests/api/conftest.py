import pytest
import httpx
from unittest.mock import patch, MagicMock
from src.api.app import create_app


@pytest.fixture
def app():
    mock_model = MagicMock()
    with patch("src.api.app.get_openai_model", return_value=mock_model):
        application = create_app()
    return application


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
