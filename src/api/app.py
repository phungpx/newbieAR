from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="newbieAR API", version="1.0.0", lifespan=lifespan)
    return app


app = create_app()
