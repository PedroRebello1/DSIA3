"""Rotas do catálogo de jogos."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, Limit, Offset
from app.schemas.game import GameCreate, GameOut
from app.services import games as service

router = APIRouter(prefix="/games", tags=["games"])

Search = Annotated[str | None, Query(max_length=100, description="Trecho do nome do jogo.")]


@router.post("", response_model=GameOut, status_code=status.HTTP_201_CREATED)
def create_game(data: GameCreate, db: DbSession) -> GameOut:
    """Cadastra um jogo no catálogo."""
    return service.create_game(db, data)


@router.get("", response_model=list[GameOut])
def list_games(
    db: DbSession, search: Search = None, limit: Limit = 20, offset: Offset = 0
) -> list[GameOut]:
    """Lista ou busca jogos pelo nome."""
    return service.list_games(db, search, limit, offset)


@router.get("/{game_id}", response_model=GameOut)
def get_game(game_id: int, db: DbSession) -> GameOut:
    """Detalhe de um jogo."""
    return service.get_game(db, game_id)
