"""Rota de recomendações."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbSession, Recommender
from app.core.config import get_settings
from app.schemas.error import error_responses
from app.schemas.recommendation import RecommendationsOut

router = APIRouter(tags=["recommendations"])

K = Annotated[int | None, Query(ge=1, le=100, description="Quantos jogos recomendar.")]


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationsOut,
    responses=error_responses(404, 422),
)
def recommend(user_id: int, db: DbSession, service: Recommender, k: K = None) -> RecommendationsOut:
    """Top-k jogos para o usuário, sem os que ele já tem ou marcou como "não gostei"."""
    return service.recommend(db, user_id, k or get_settings().default_k)
