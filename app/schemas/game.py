"""Schemas de jogos."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

GameName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Tag = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=50)
]


class GameCreate(BaseModel):
    """Dados para cadastrar um jogo."""

    name: GameName = Field(description="Nome do jogo (único, sem diferenciar maiúsculas).")
    tags: list[Tag] = Field(
        default_factory=list, max_length=20, description="Gêneros ou palavras-chave opcionais."
    )


class GameOut(BaseModel):
    """Jogo do catálogo."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    tags: list[str] = Field(description="Gêneros ou palavras-chave (pode ser vazio).")
    created_at: datetime = Field(description="Data de cadastro.")
