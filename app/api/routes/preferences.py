"""Rotas de preferências do usuário (compras, horas jogadas e "não gostei")."""

from fastapi import APIRouter

from app.api.deps import DbSession, UserId
from app.schemas.error import error_responses
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.schemas.user import UserDetail
from app.services import preferences as service

router = APIRouter(prefix="/users/{user_id}/preferences", tags=["preferences"])


@router.put(
    "",
    response_model=UserDetail,
    responses=error_responses(404, 422),
    summary="Substitui as preferências",
)
def replace_preferences(user_id: UserId, data: PreferencesReplace, db: DbSession) -> UserDetail:
    """Substitui todo o histórico do usuário pela lista enviada.

    Jogos fora da lista são removidos; uma lista vazia limpa o histórico.
    """
    return service.replace_preferences(db, user_id, data)


@router.patch(
    "",
    response_model=UserDetail,
    responses=error_responses(404, 422),
    summary="Atualiza as preferências",
)
def update_preferences(user_id: UserId, data: PreferencesPatch, db: DbSession) -> UserDetail:
    """Adiciona, atualiza ou remove jogos do histórico.

    - `upsert`: cria o jogo no histórico ou altera só os campos enviados.
    - `remove`: tira jogos do histórico (remover um jogo ausente não é erro).
    - Para marcar "não gostei", envie `{"game_id": ..., "disliked": true}` em `upsert`.
    """
    return service.update_preferences(db, user_id, data)
