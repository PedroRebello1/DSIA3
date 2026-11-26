"""Ponto de entrada da API FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.openapi import DESCRIPTION, TAGS
from app.api.routes import games, health, model, preferences, recommendations, users
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.services.recommender_service import RecommenderService

logger = logging.getLogger(__name__)


def load_model(path) -> RecommenderService | None:
    """Carrega o artefato do híbrido, se existir; ``None`` em caso de ausência ou erro."""
    if not path.exists():
        logger.warning("Modelo não encontrado em %s; usando popularidade até o retreino.", path)
        return None
    try:
        service = RecommenderService.from_artifact(path)
    except Exception:
        logger.exception("Falha ao carregar o modelo de %s", path)
        return None
    logger.info("Modelo %s carregado (versão %s).", service.strategy, service.version)
    return service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo híbrido no startup e o libera no shutdown.

    Sem artefato, o serviço de recomendação é criado no primeiro pedido
    (``get_recommender_service``), com popularidade calculada a partir do banco.
    """
    app.state.recommender = load_model(get_settings().model_path)
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
    application.include_router(model.router)
    return application


app = create_app()
