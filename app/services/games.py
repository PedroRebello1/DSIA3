"""Regras de negócio do catálogo de jogos."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.db import crud
from app.db.models import Game
from app.schemas.game import GameCreate


def create_game(db: Session, data: GameCreate) -> Game:
    """Cadastra um jogo; o nome não pode repetir (sem diferenciar maiúsculas)."""
    existing = crud.get_game_by_name(db, data.name)
    if existing is not None:
        raise ConflictError(f"Já existe o jogo '{existing.name}' (id {existing.id})")
    try:
        game = crud.create_game(db, data.name, data.tags)
        db.commit()
    except IntegrityError as exc:  # corrida entre duas requisições com o mesmo nome
        db.rollback()
        raise ConflictError(f"Já existe um jogo com o nome '{data.name}'") from exc
    return game


def get_game(db: Session, game_id: int) -> Game:
    """Jogo pelo ID; 404 se não existir."""
    game = crud.get_game(db, game_id)
    if game is None:
        raise NotFoundError(f"Jogo {game_id} não encontrado")
    return game


def list_games(db: Session, search: str | None, limit: int, offset: int) -> list[Game]:
    """Página do catálogo, opcionalmente filtrada por trecho do nome."""
    return crud.list_games(db, search=search, limit=limit, offset=offset)
