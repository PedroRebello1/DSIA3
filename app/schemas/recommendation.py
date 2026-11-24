"""Schemas das recomendações."""

from pydantic import BaseModel, Field


class RecommendedGame(BaseModel):
    """Um jogo recomendado."""

    game_id: int
    name: str
    score: float = Field(description="Pontuação do modelo (maior é melhor).")


class RecommendationsOut(BaseModel):
    """Lista de recomendações para um usuário."""

    user_id: int
    strategy: str = Field(description="Estratégia usada para gerar a lista.")
    model_version: str = Field(description="Data/hora em que o modelo foi treinado.")
    recommendations: list[RecommendedGame]
