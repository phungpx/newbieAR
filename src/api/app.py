from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.ingestion.router import router as ingestion_router
from src.api.retrieval.router import router as retrieval_router
from src.api.agents.router import router as agents_router
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="newbieAR API", version="1.0.0", lifespan=lifespan)
    prefix = settings.api_prefix  # "/api/v1"
    app.include_router(ingestion_router, prefix=prefix)
    app.include_router(retrieval_router, prefix=prefix)
    app.include_router(agents_router, prefix=prefix)
    return app


app = create_app()
