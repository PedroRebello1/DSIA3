"""Schemas das interações usuário–jogo (histórico e preferências)."""

from datetime import datetime

from pydantic import BaseModel, Field

MAX_HOURS = 100_000.0


class InteractionIn(BaseModel):
    """Um jogo no histórico do usuário."""

    game_id: int = Field(gt=0, description="ID do jogo.")
    purchased: bool = Field(True, description="Se o usuário comprou o jogo.")
    hours: float = Field(0.0, ge=0, le=MAX_HOURS, description="Horas jogadas.")
    disliked: bool = Field(False, description='Marcado como "não gostei".')


class InteractionOut(BaseModel):
    """Jogo do histórico, com o nome para facilitar a leitura."""

    game_id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    purchased: bool = Field(description="Se o usuário comprou o jogo.")
    hours: float = Field(description="Horas jogadas.")
    disliked: bool = Field(description='Marcado como "não gostei".')
    updated_at: datetime = Field(description="Última alteração.")
