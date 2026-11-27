"""Regras de negócio de usuários."""

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.db import crud
from app.db.models import Interaction, User
from app.schemas.interaction import InteractionIn, InteractionOut
from app.schemas.user import UserCreate, UserDetail, UserOut


def require_user(db: Session, user_id: int) -> User:
    """Usuário pelo ID; 404 se não existir."""
    user = crud.get_user(db, user_id)
    if user is None:
        raise NotFoundError(f"Usuário {user_id} não encontrado")
    return user


def require_games(db: Session, game_ids: list[int]) -> None:
    """Garante que todos os jogos existem; 404 com a lista dos que faltam."""
    missing = crud.missing_game_ids(db, game_ids)
    if missing:
        raise NotFoundError(f"Jogos não encontrados: {sorted(missing)}")


def to_interaction_out(interaction: Interaction) -> InteractionOut:
    """Interação do banco no formato da resposta, com o nome do jogo."""
    return InteractionOut(
        game_id=interaction.game_id,
        name=interaction.game.name,
        purchased=interaction.purchased,
        hours=interaction.hours,
        disliked=interaction.disliked,
        updated_at=interaction.updated_at,
    )


def user_detail(db: Session, user: User) -> UserDetail:
    """Usuário com o histórico completo."""
    history = [to_interaction_out(i) for i in crud.get_interactions(db, user.id)]
    return UserDetail(**UserOut.model_validate(user).model_dump(), history=history)


def add_interactions(db: Session, user_id: int, items: list[InteractionIn]) -> None:
    """Cria ou atualiza as interações do usuário (sem ``commit``)."""
    for item in items:
        crud.upsert_interaction(db, user_id, **item.model_dump())


def create_user(db: Session, data: UserCreate) -> UserDetail:
    """Cria o usuário e, se vierem, os jogos iniciais do histórico."""
    if data.steam_real_id is not None and crud.get_user_by_steam_id(db, data.steam_real_id):
        raise ConflictError(f"Já existe usuário com steam_real_id {data.steam_real_id}")
    require_games(db, [g.game_id for g in data.games])

    user = crud.create_user(db, data.steam_real_id)
    add_interactions(db, user.id, data.games)
    db.commit()
    return user_detail(db, user)


def get_user(db: Session, user_id: int) -> UserDetail:
    """Usuário e histórico; 404 se não existir."""
    return user_detail(db, require_user(db, user_id))


def list_users(db: Session, limit: int, offset: int) -> list[User]:
    """Página de usuários, por ID."""
    return crud.list_users(db, limit=limit, offset=offset)
