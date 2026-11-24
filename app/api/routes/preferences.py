"""Rotas de preferências do usuário (compras, horas jogadas e "não gostei")."""

from fastapi import APIRouter

from app.api.deps import DbSession
from app.schemas.error import error_responses
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.schemas.user import UserDetail
from app.services import preferences as service

router = APIRouter(prefix="/users/{user_id}/preferences", tags=["preferences"])


@router.put("", response_model=UserDetail, responses=error_responses(404, 422))
def replace_preferences(user_id: int, data: PreferencesReplace, db: DbSession) -> UserDetail:
    """Substitui todo o histórico do usuário."""
    return service.replace_preferences(db, user_id, data)


@router.patch("", response_model=UserDetail, responses=error_responses(404, 422))
def update_preferences(user_id: int, data: PreferencesPatch, db: DbSession) -> UserDetail:
    """Adiciona, atualiza ou remove jogos do histórico."""
    return service.update_preferences(db, user_id, data)
