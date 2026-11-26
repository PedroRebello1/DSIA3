"""Rotas de informações e manutenção do modelo."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query, Request, status

from app.api.deps import DbSession, LoadedRecommender
from app.core.config import get_settings
from app.schemas.error import error_responses
from app.schemas.model import ModelInfoOut, RetrainOut
from app.services.model_info import get_model_info
from app.services.model_training import is_retraining, retrain_model, start_retrain

router = APIRouter(prefix="/model", tags=["model"])

Search = Annotated[
    bool, Query(description="Faz a busca de hiperparâmetros (mais lento, alguns minutos).")
]


@router.get(
    "/info",
    response_model=ModelInfoOut,
    responses=error_responses(),
    summary="Informações do modelo",
)
def model_info(request: Request, db: DbSession, service: LoadedRecommender) -> ModelInfoOut:
    """Modelo em memória (estratégia e versão), último treino registrado e se há retreino
    em andamento.

    Não força o carregamento do modelo: sem artefato treinado, `loaded` é `false` até o
    primeiro pedido de recomendação.
    """
    return get_model_info(db, service, retraining=is_retraining(request.app))


@router.post(
    "/retrain",
    response_model=RetrainOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses=error_responses(409, 422),
    summary="Retreina o modelo em segundo plano",
)
def retrain(
    request: Request, db: DbSession, background: BackgroundTasks, search: Search = False
) -> RetrainOut:
    """Treina o híbrido com os dados atuais do banco, sem interromper a API.

    Ao terminar, o novo modelo substitui o atual de uma vez (troca atômica) e o treino é
    registrado em `model_runs`. Acompanhe por `GET /model/info` (`retraining` e
    `version`). Só um retreino roda por vez (409 se já houver um).
    """
    start_retrain(request.app)
    background.add_task(
        retrain_model, request.app, db.get_bind(), get_settings().artifacts_dir, search
    )
    return RetrainOut(status="started", message="Retreino iniciado; acompanhe em GET /model/info.")
