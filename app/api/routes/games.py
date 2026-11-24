"""Rotas do catálogo de jogos."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, GameId, Limit, Offset
from app.schemas.error import error_responses
from app.schemas.game import GameCreate, GameOut
from app.services import games as service

router = APIRouter(prefix="/games", tags=["games"])

Search = Annotated[str | None, Query(max_length=100, description="Trecho do nome do jogo.")]


@router.post(
    "",
    response_model=GameOut,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409, 422),
    summary="Cadastra um jogo",
)
def create_game(data: GameCreate, db: DbSession) -> GameOut:
    """Cadastra um jogo no catálogo.

    O nome é único sem diferenciar maiúsculas (409 se já existir). As `tags` são
    opcionais e ajudam o modelo de conteúdo a encontrar jogos parecidos.
    """
    return service.create_game(db, data)


@router.get(
    "", response_model=list[GameOut], responses=error_responses(422), summary="Lista ou busca jogos"
)
def list_games(
    db: DbSession, search: Search = None, limit: Limit = 20, offset: Offset = 0
) -> list[GameOut]:
    """Lista jogos por ID; com `search`, filtra por trecho do nome (sem diferenciar maiúsculas)."""
    return service.list_games(db, search, limit, offset)


@router.get(
    "/{game_id}",
    response_model=GameOut,
    responses=error_responses(404, 422),
    summary="Consulta um jogo",
)
def get_game(game_id: GameId, db: DbSession) -> GameOut:
    """Detalhe de um jogo."""
    return service.get_game(db, game_id)
