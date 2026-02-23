import time

from fastapi import APIRouter

from src.api.schemas import ModelInfo, ModelsResponse
from src.api.services.agent_runner import KNOWN_MODELS

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    now = int(time.time())
    return ModelsResponse(
        data=[ModelInfo(id=m, created=now) for m in KNOWN_MODELS]
    )
