"""Rotas de informações e manutenção do modelo."""

from fastapi import APIRouter

from app.api.deps import DbSession, LoadedRecommender
from app.schemas.error import error_responses
from app.schemas.model import ModelInfoOut
from app.services.model_info import get_model_info

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "/info",
    response_model=ModelInfoOut,
    responses=error_responses(),
    summary="Informações do modelo",
)
def model_info(db: DbSession, service: LoadedRecommender) -> ModelInfoOut:
    """Modelo em memória (estratégia e versão) e o último treino registrado, com as métricas.

    Não força o carregamento do modelo: `loaded` é `false` até o primeiro uso.
    """
    return get_model_info(db, service)
