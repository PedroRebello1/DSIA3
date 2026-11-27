"""Rotas de preferências do usuário (compras, horas jogadas e "não gostei")."""

from typing import Annotated

from fastapi import APIRouter, Body

from app.api.deps import DbSession, UserId
from app.schemas.error import error_responses
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.schemas.user import UserDetail
from app.services import preferences as service

router = APIRouter(prefix="/users/{user_id}/preferences", tags=["preferences"])

ReplaceBody = Annotated[
    PreferencesReplace,
    Body(
        openapi_examples={
            "substituir": {
                "summary": "Novo histórico completo",
                "value": {
                    "games": [
                        {"game_id": 2, "hours": 95},
                        {"game_id": 4, "hours": 12.5},
                        {"game_id": 1, "disliked": True},
                    ]
                },
            },
            "limpar": {"summary": "Apaga todo o histórico", "value": {"games": []}},
        }
    ),
]
PatchBody = Annotated[
    PreferencesPatch,
    Body(
        openapi_examples={
            "adicionar": {
                "summary": "Adiciona um jogo e atualiza as horas de outro",
                "value": {"upsert": [{"game_id": 4, "hours": 12.5}, {"game_id": 2, "hours": 95}]},
            },
            "nao_gostei": {
                "summary": 'Marca um jogo como "não gostei"',
                "value": {"upsert": [{"game_id": 1, "disliked": True}]},
            },
            "remover": {"summary": "Remove jogos do histórico", "value": {"remove": [43]}},
        }
    ),
]


@router.put(
    "",
    response_model=UserDetail,
    responses=error_responses(404, 422, messages={404: "Jogos não encontrados: [99999]"}),
    summary="Substitui as preferências",
)
def replace_preferences(user_id: UserId, data: ReplaceBody, db: DbSession) -> UserDetail:
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
def update_preferences(user_id: UserId, data: PatchBody, db: DbSession) -> UserDetail:
    """Adiciona, atualiza ou remove jogos do histórico.

    - `upsert`: cria o jogo no histórico ou altera só os campos enviados.
    - `remove`: tira jogos do histórico (remover um jogo ausente não é erro).
    - Para marcar "não gostei", envie `{"game_id": ..., "disliked": true}` em `upsert`.
    """
    return service.update_preferences(db, user_id, data)
