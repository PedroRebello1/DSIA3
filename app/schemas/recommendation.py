"""Schemas das recomendações."""

from pydantic import BaseModel, Field


class RecommendedGame(BaseModel):
    """Um jogo recomendado."""

    game_id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    score: float = Field(description="Pontuação do modelo (maior é melhor).")


class RecommendationsOut(BaseModel):
    """Lista de recomendações para um usuário."""

    user_id: int = Field(description="ID do usuário.")
    strategy: str = Field(description="Estratégia usada para gerar a lista.")
    model_version: str = Field(description="Data/hora em que o modelo foi treinado.")
    recommendations: list[RecommendedGame] = Field(description="Jogos, do mais indicado ao menos.")
