"""Ponto de entrada da API FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.openapi import DESCRIPTION, TAGS
from app.api.routes import games, health, preferences, recommendations, users
from app.core.config import get_settings
from app.core.errors import register_error_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepara recursos compartilhados no startup e os libera no shutdown.

    O serviço de recomendação é criado no primeiro pedido (``get_recommender_service``),
    quando já existe uma sessão com o banco.
    """
    app.state.recommender = None
    yield
    app.state.recommender = None


def create_app() -> FastAPI:
    """Cria a aplicação com as rotas registradas."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    register_error_handlers(application)
    application.include_router(health.router)
    application.include_router(users.router)
    application.include_router(preferences.router)
    application.include_router(recommendations.router)
    application.include_router(games.router)
    return application


app = create_app()
