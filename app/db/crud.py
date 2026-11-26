"""Repositórios: acesso ao banco, sem regras de negócio.

As funções dão ``flush`` mas não ``commit``; quem chama controla a transação.
"""

from collections.abc import Iterable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import Game, Interaction, ModelRun, User

# ---- Usuários ----


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def get_user_by_steam_id(db: Session, steam_real_id: int) -> User | None:
    return db.scalar(select(User).where(User.steam_real_id == steam_real_id))


def list_users(db: Session, limit: int = 20, offset: int = 0) -> list[User]:
    return list(db.scalars(select(User).order_by(User.id).limit(limit).offset(offset)))


def create_user(db: Session, steam_real_id: int | None = None) -> User:
    user = User(steam_real_id=steam_real_id)
    db.add(user)
    db.flush()
    return user


# ---- Jogos ----


def get_game(db: Session, game_id: int) -> Game | None:
    return db.get(Game, game_id)


def get_game_by_name(db: Session, name: str) -> Game | None:
    """Busca pelo nome exato, sem diferenciar maiúsculas."""
    return db.scalar(select(Game).where(func.lower(Game.name) == name.strip().lower()))


def list_games(
    db: Session, search: str | None = None, limit: int = 20, offset: int = 0
) -> list[Game]:
    """Lista jogos por ID; ``search`` filtra por trecho do nome (sem diferenciar maiúsculas)."""
    stmt = select(Game)
    if search:
        stmt = stmt.where(Game.name.ilike(f"%{search.strip()}%"))
    return list(db.scalars(stmt.order_by(Game.id).limit(limit).offset(offset)))


def create_game(db: Session, name: str, tags: list[str] | None = None) -> Game:
    game = Game(name=name, tags=list(tags or []))
    db.add(game)
    db.flush()
    return game


def get_games_by_ids(db: Session, game_ids: Iterable[int]) -> dict[int, Game]:
    ids = set(game_ids)
    if not ids:
        return {}
    return {g.id: g for g in db.scalars(select(Game).where(Game.id.in_(ids)))}


def missing_game_ids(db: Session, game_ids: Iterable[int]) -> set[int]:
    """IDs da lista que não existem no catálogo."""
    ids = set(game_ids)
    return ids - set(get_games_by_ids(db, ids))


# ---- Interações ----


def get_interactions(db: Session, user_id: int) -> list[Interaction]:
    """Histórico do usuário, com o jogo já carregado."""
    stmt = (
        select(Interaction)
        .options(joinedload(Interaction.game))
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.game_id)
    )
    return list(db.scalars(stmt))


def upsert_interaction(db: Session, user_id: int, game_id: int, **fields) -> Interaction:
    """Cria a interação ou atualiza só os campos informados (valores ``None`` são ignorados)."""
    values = {k: v for k, v in fields.items() if v is not None}
    interaction = db.get(Interaction, (user_id, game_id))
    if interaction is None:
        interaction = Interaction(user_id=user_id, game_id=game_id, **values)
        db.add(interaction)
    else:
        for key, value in values.items():
            setattr(interaction, key, value)
    db.flush()
    return interaction


def delete_interactions(db: Session, user_id: int, game_ids: Iterable[int] | None = None) -> int:
    """Remove interações do usuário (todas, se ``game_ids`` for ``None``)."""
    stmt = delete(Interaction).where(Interaction.user_id == user_id)
    if game_ids is not None:
        stmt = stmt.where(Interaction.game_id.in_(set(game_ids)))
    result = db.execute(stmt)
    db.expire_all()
    return result.rowcount


# ---- Treinos do modelo ----


def get_latest_model_run(db: Session) -> ModelRun | None:
    """Treino mais recente registrado em ``model_runs``."""
    return db.scalar(select(ModelRun).order_by(ModelRun.trained_at.desc(), ModelRun.id.desc()))
