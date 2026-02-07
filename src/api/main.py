from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger

from src.api.routers import ingestion, retrieval, agents
from src.api.middleware.error_handler import global_exception_handler
from src.api.services import session_manager
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting FastAPI application...")

    # Startup tasks
    logger.info(f"API listening on {settings.api.api_host}:{settings.api.api_port}")
    logger.info(f"API prefix: {settings.api.api_prefix}")

    yield

    # Shutdown tasks
    logger.info("Shutting down FastAPI application...")
    session_manager.cleanup_expired()


app = FastAPI(
    title="newbieAR API",
    description="REST APIs for Ingestion, Retrieval, and Agentic RAG",
    version="1.0.0",
    docs_url=f"{settings.api.api_prefix}/docs",
    redoc_url=f"{settings.api.api_prefix}/redoc",
    openapi_url=f"{settings.api.api_prefix}/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(Exception, global_exception_handler)

# Include routers
app.include_router(ingestion.router, prefix=settings.api.api_prefix)
app.include_router(retrieval.router, prefix=settings.api.api_prefix)
app.include_router(agents.router, prefix=settings.api.api_prefix)


@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    """Redirect /docs to prefixed docs URL"""
    return RedirectResponse(url=f"{settings.api.api_prefix}/docs")


@app.get("/redoc", include_in_schema=False)
async def redoc_redirect():
    """Redirect /redoc to prefixed redoc URL"""
    return RedirectResponse(url=f"{settings.api.api_prefix}/redoc")


@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "newbieAR API"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check - verify DB connections"""
    try:
        from src.deps import QdrantVectorStore

        # Test Qdrant connection
        vector_store = QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
        vector_store.list_collections()

        return {"status": "ready", "checks": {"qdrant": "connected"}}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=True,
        log_level="info",
    )
