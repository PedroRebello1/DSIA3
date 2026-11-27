"""Schemas das recomendações."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Strategy(StrEnum):
    """Estratégias de recomendação disponíveis."""

    hybrid = "hybrid"
    cf = "cf"
    content = "content"
    popular = "popular"


RECOMMENDATIONS_EXAMPLE = [
    {
        "game_id": 4,
        "name": "Fallout New Vegas",
        "score": 0.784,
        "reason": "Porque você jogou Fallout 4",
    },
    {
        "game_id": 511,
        "name": "The Witcher 2 Assassins of Kings Enhanced Edition",
        "score": 0.757,
        "reason": "Porque você jogou The Witcher 3 Wild Hunt",
    },
    {
        "game_id": 1,
        "name": "The Elder Scrolls V Skyrim",
        "score": 0.667,
        "reason": "Porque você jogou Fallout 4",
    },
]
SIMILAR_EXAMPLE = [
    {
        "game_id": 4,
        "name": "Fallout New Vegas",
        "score": 0.77,
        "reason": "Quem jogou Fallout 4 também jogou",
    },
    {
        "game_id": 31,
        "name": "Fallout New Vegas Dead Money",
        "score": 0.556,
        "reason": "Quem jogou Fallout 4 também jogou",
    },
    {
        "game_id": 32,
        "name": "Fallout New Vegas Honest Hearts",
        "score": 0.542,
        "reason": "Quem jogou Fallout 4 também jogou",
    },
]


class RecommendedGame(BaseModel):
    """Um jogo recomendado."""

    model_config = ConfigDict(json_schema_extra={"examples": RECOMMENDATIONS_EXAMPLE[:1]})

    game_id: int = Field(description="ID do jogo.")
    name: str = Field(description="Nome do jogo.")
    score: float = Field(description="Pontuação do modelo (maior é melhor).")
    reason: str | None = Field(None, description="Por que o jogo foi recomendado.")


class RecommendationsOut(BaseModel):
    """Lista de recomendações para um usuário."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "user_id": 12394,
                    "strategy": "hybrid",
                    "model_version": "2026-11-27T14:00:00+00:00",
                    "recommendations": RECOMMENDATIONS_EXAMPLE,
                }
            ]
        }
    )

    user_id: int = Field(description="ID do usuário.")
    strategy: Strategy = Field(description="Estratégia usada para gerar a lista.")
    model_version: str = Field(description="Data/hora em que o modelo foi treinado.")
    recommendations: list[RecommendedGame] = Field(description="Jogos, do mais indicado ao menos.")


class SimilarGamesOut(BaseModel):
    """Jogos parecidos com um jogo de referência."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"game_id": 2, "name": "Fallout 4", "similar": SIMILAR_EXAMPLE}]
        }
    )

    game_id: int = Field(description="ID do jogo de referência.")
    name: str = Field(description="Nome do jogo de referência.")
    similar: list[RecommendedGame] = Field(description="Jogos parecidos, do mais ao menos.")
