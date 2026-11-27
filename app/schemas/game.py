"""Schemas de jogos."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

GameName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Tag = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=50)
]

GAME_EXAMPLE = {
    "id": 5156,
    "name": "Hades",
    "tags": ["roguelike", "ação", "indie"],
    "created_at": "2026-11-27T14:00:00",
}


class GameCreate(BaseModel):
    """Dados para cadastrar um jogo."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"name": "Hades", "tags": ["Roguelike", "Ação", "Indie"]}]}
    )

    name: GameName = Field(description="Nome do jogo (único, sem diferenciar maiúsculas).")
    tags: list[Tag] = Field(
        default_factory=list, max_length=20, description="Gêneros ou palavras-chave opcionais."
    )

    @field_validator("tags")
    @classmethod
    def drop_repeated_tags(cls, tags: list[str]) -> list[str]:
        """Remove tags repetidas (depois de normalizadas), mantendo a ordem."""
        return list(dict.fromkeys(tags))


class GameOut(BaseModel):
    """Jogo do catálogo."""

    model_config = ConfigDict(from_attributes=True, json_schema_extra={"examples": [GAME_EXAMPLE]})

    id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    tags: list[str] = Field(description="Gêneros ou palavras-chave (pode ser vazio).")
    created_at: datetime = Field(description="Data de cadastro.")
