from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.models import get_openai_model
from src.api.routers.sessions import router as sessions_router
from src.api.routers.chat import router as chat_router
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = get_openai_model(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions_router, prefix=settings.api_prefix)
    app.include_router(chat_router, prefix=settings.api_prefix)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
