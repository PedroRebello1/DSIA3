"""Rotas do catálogo de jogos."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import DbSession, GameId, K, Limit, LoadedRecommender, Offset, Recommender
from app.core.config import get_settings
from app.schemas.error import error_responses
from app.schemas.game import GameCreate, GameOut
from app.schemas.recommendation import SimilarGamesOut
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
def create_game(data: GameCreate, db: DbSession, model: LoadedRecommender) -> GameOut:
    """Cadastra um jogo no catálogo.

    O nome é único sem diferenciar maiúsculas (409 se já existir). As `tags` são
    opcionais e ajudam o modelo de conteúdo a encontrar jogos parecidos. O jogo já entra
    no modelo em memória: aparece em `/similar` e pode ser recomendado sem retreino.
    """
    game = service.create_game(db, data)
    if model is not None:
        model.add_game(game.id, game.name, game.tags)
    return game


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


@router.get(
    "/{game_id}/similar",
    response_model=SimilarGamesOut,
    responses=error_responses(404, 422),
    summary="Lista jogos parecidos",
)
def similar_games(
    game_id: GameId, db: DbSession, model: Recommender, k: K = None
) -> SimilarGamesOut:
    """Jogos parecidos com o informado, pelo título e pelas tags (modelo de conteúdo)."""
    return model.similar_games(db, game_id, k or get_settings().default_k)
