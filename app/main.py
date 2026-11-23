"""Ponto de entrada da API FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepara recursos compartilhados no startup e os libera no shutdown."""
    app.state.model = None
    yield
    app.state.model = None


def create_app() -> FastAPI:
    """Cria a aplicação com as rotas registradas."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="API de recomendação de jogos da Steam.",
        lifespan=lifespan,
    )
    application.include_router(health.router)
    return application


app = create_app()
