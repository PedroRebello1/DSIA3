"""Regras de atualização das preferências do usuário."""

from sqlalchemy.orm import Session

from app.db import crud
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.schemas.user import UserDetail
from app.services.users import add_interactions, require_games, require_user, user_detail


def replace_preferences(db: Session, user_id: int, data: PreferencesReplace) -> UserDetail:
    """Troca todo o histórico do usuário pelo conjunto enviado."""
    user = require_user(db, user_id)
    require_games(db, [g.game_id for g in data.games])

    crud.delete_interactions(db, user_id)
    add_interactions(db, user_id, data.games)
    db.commit()
    return user_detail(db, user)


def update_preferences(db: Session, user_id: int, data: PreferencesPatch) -> UserDetail:
    """Adiciona/atualiza os jogos de ``upsert`` e remove os de ``remove``.

    Remover um jogo que não está no histórico não é erro (operação idempotente).
    """
    user = require_user(db, user_id)
    require_games(db, [g.game_id for g in data.upsert])

    for item in data.upsert:
        crud.upsert_interaction(db, user_id, **item.model_dump())
    if data.remove:
        crud.delete_interactions(db, user_id, data.remove)
    db.commit()
    return user_detail(db, user)
