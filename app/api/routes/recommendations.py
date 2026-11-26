"""Rota de recomendações."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbSession, K, Recommender, UserId
from app.core.config import get_settings
from app.schemas.error import error_responses
from app.schemas.recommendation import RecommendationsOut, Strategy

router = APIRouter(tags=["recommendations"])

StrategyParam = Annotated[
    Strategy | None, Query(description="Estratégia; padrão: a do modelo carregado.")
]
ExcludeOwned = Annotated[bool, Query(description="Exclui os jogos que o usuário já tem.")]


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationsOut,
    responses=error_responses(404, 422, 503),
    summary="Recomenda jogos para um usuário",
)
def recommend(
    user_id: UserId,
    db: DbSession,
    service: Recommender,
    k: K = None,
    strategy: StrategyParam = None,
    exclude_owned: ExcludeOwned = True,
) -> RecommendationsOut:
    """Top-k jogos para o usuário, com o motivo de cada recomendação (`reason`).

    - `strategy`: `hybrid` (padrão quando o modelo treinado está carregado), `cf`
      (item-kNN), `content` (título/tags) ou `popular`. Sem o modelo treinado, só
      `popular` está disponível (503 nas demais).
    - `exclude_owned`: se `false`, jogos do histórico podem aparecer. Jogos marcados como
      "não gostei" nunca aparecem.
    - `k`: se omitido, usa o padrão da configuração (`DEFAULT_K`).
    """
    return service.recommend(
        db, user_id, k or get_settings().default_k, strategy=strategy, exclude_owned=exclude_owned
    )
