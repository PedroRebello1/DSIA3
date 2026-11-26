"""Rota de recomendações."""

from fastapi import APIRouter

from app.api.deps import DbSession, K, Recommender, UserId
from app.core.config import get_settings
from app.schemas.error import error_responses
from app.schemas.recommendation import RecommendationsOut

router = APIRouter(tags=["recommendations"])


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationsOut,
    responses=error_responses(404, 422),
    summary="Recomenda jogos para um usuário",
)
def recommend(
    user_id: UserId, db: DbSession, service: Recommender, k: K = None
) -> RecommendationsOut:
    """Top-k jogos para o usuário, ordenados pelo score do modelo.

    Jogos que o usuário já tem ou marcou como "não gostei" ficam de fora. Se `k` não
    for informado, usa o padrão da configuração (`DEFAULT_K`).
    """
    return service.recommend(db, user_id, k or get_settings().default_k)
