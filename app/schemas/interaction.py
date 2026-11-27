"""Schemas das interações usuário–jogo (histórico e preferências)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

MAX_HOURS = 100_000.0

HISTORY_EXAMPLE = [
    {
        "game_id": 2,
        "name": "Fallout 4",
        "purchased": True,
        "hours": 80.5,
        "disliked": False,
        "updated_at": "2026-11-27T14:00:00",
    },
    {
        "game_id": 43,
        "name": "Sid Meier's Civilization V",
        "purchased": True,
        "hours": 30.0,
        "disliked": False,
        "updated_at": "2026-11-27T14:00:00",
    },
    {
        "game_id": 512,
        "name": "The Witcher 3 Wild Hunt",
        "purchased": True,
        "hours": 120.0,
        "disliked": False,
        "updated_at": "2026-11-27T14:00:00",
    },
]


class InteractionIn(BaseModel):
    """Um jogo no histórico do usuário."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"game_id": 2, "purchased": True, "hours": 80.5, "disliked": False}]
        }
    )

    game_id: int = Field(gt=0, description="ID do jogo.")
    purchased: bool = Field(True, description="Se o usuário comprou o jogo.")
    hours: float = Field(0.0, ge=0, le=MAX_HOURS, description="Horas jogadas.")
    disliked: bool = Field(False, description='Marcado como "não gostei".')


class InteractionOut(BaseModel):
    """Jogo do histórico, com o nome para facilitar a leitura."""

    model_config = ConfigDict(json_schema_extra={"examples": HISTORY_EXAMPLE[:1]})

    game_id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    purchased: bool = Field(description="Se o usuário comprou o jogo.")
    hours: float = Field(description="Horas jogadas.")
    disliked: bool = Field(description='Marcado como "não gostei".')
    updated_at: datetime = Field(description="Última alteração.")
