"""Schemas de usuários."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.interaction import InteractionIn, InteractionOut


def ensure_unique_games[T: BaseModel](items: list[T]) -> list[T]:
    """Rejeita listas com o mesmo jogo (``game_id``) repetido."""
    ids = [item.game_id for item in items]
    repeated = sorted({g for g in ids if ids.count(g) > 1})
    if repeated:
        raise ValueError(f"jogos repetidos na lista: {repeated}")
    return items


class UserCreate(BaseModel):
    """Dados para criar um usuário, opcionalmente já com jogos no histórico."""

    steam_real_id: int | None = Field(None, gt=0, description="ID original da Steam, se houver.")
    games: list[InteractionIn] = Field(default_factory=list, description="Histórico inicial.")

    _unique_games = field_validator("games")(ensure_unique_games)


class UserOut(BaseModel):
    """Usuário (sem o histórico)."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="ID do usuário.")
    steam_real_id: int | None = Field(description="ID original da Steam (nulo se criado pela API).")
    created_at: datetime = Field(description="Data de cadastro.")


class UserDetail(UserOut):
    """Usuário com o histórico de jogos."""

    history: list[InteractionOut] = Field(description="Jogos do usuário, por ID.")
