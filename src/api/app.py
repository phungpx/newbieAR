from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.retrieval.basic_rag import BasicRAG
from src.retrieval.graph_rag import GraphRetrieval
from src.settings import settings
from src.api.routers import chat, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=False
    )
    app.state.basic_rag = BasicRAG()
    app.state.graph_retrieval = GraphRetrieval()
    app.state.session_ttl = settings.session_ttl_hours * 3600
    yield
    # Shutdown
    await app.state.redis.aclose()
    await app.state.graph_retrieval.close()


app = FastAPI(title="newbieAR Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(chat.router)
