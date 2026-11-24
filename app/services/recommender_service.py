"""Ponte entre a API e o pacote ``recsys``.

O modelo recebe só o histórico (``{game_id: score}``); este serviço monta esse histórico
a partir do banco, chama o modelo e devolve a resposta com os nomes dos jogos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import crud
from app.db.models import Game, Interaction
from app.schemas.recommendation import RecommendationsOut, RecommendedGame
from app.services.users import require_user
from recsys.base import Recommender
from recsys.data import DEFAULT_ALPHA, confidence_score
from recsys.models.popularity import PopularityRecommender


@dataclass
class UserProfile:
    """Histórico do usuário no formato que o modelo espera."""

    history: dict[int, float] = field(default_factory=dict)  # jogos positivos -> score
    owned: set[int] = field(default_factory=set)  # tudo que está no histórico
    disliked: set[int] = field(default_factory=set)


def build_profile(interactions: list[Interaction], alpha: float = DEFAULT_ALPHA) -> UserProfile:
    """Separa jogos positivos (com score de confiança) e jogos marcados como "não gostei"."""
    profile = UserProfile()
    for inter in interactions:
        profile.owned.add(inter.game_id)
        if inter.disliked:
            profile.disliked.add(inter.game_id)
        else:
            profile.history[inter.game_id] = float(confidence_score(inter.hours, alpha))
    return profile


def load_training_data(
    db: Session, alpha: float = DEFAULT_ALPHA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê do banco as interações positivas e o catálogo, no formato de ``recsys``."""
    rows = db.execute(
        select(Interaction.user_id, Interaction.game_id, Interaction.hours).where(
            Interaction.disliked.is_(False)
        )
    ).all()
    interactions = pd.DataFrame(rows, columns=["user_id", "game_id", "hours"])
    interactions["score"] = confidence_score(interactions["hours"].to_numpy(float), alpha)

    games = pd.DataFrame(
        db.execute(select(Game.id, Game.name, Game.tags)).all(),
        columns=["game_id", "name", "tags"],
    )
    return interactions, games


class RecommenderService:
    """Guarda o modelo em memória e atende os pedidos de recomendação."""

    def __init__(
        self, model: Recommender, strategy: str, version: str, alpha: float = DEFAULT_ALPHA
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.version = version
        self.alpha = alpha

    @classmethod
    def from_db(cls, db: Session, alpha: float = DEFAULT_ALPHA) -> RecommenderService:
        """Treina o ``PopularityRecommender`` com os dados atuais do banco."""
        interactions, games = load_training_data(db, alpha)
        model = PopularityRecommender()
        model.fit(interactions, games)
        version = datetime.now(UTC).isoformat(timespec="seconds")
        return cls(model, strategy="popular", version=version, alpha=alpha)

    def recommend(self, db: Session, user_id: int, k: int) -> RecommendationsOut:
        """Recomenda ``k`` jogos, sem repetir os que já estão no histórico do usuário."""
        require_user(db, user_id)
        profile = build_profile(crud.get_interactions(db, user_id), self.alpha)
        recs = self.model.recommend(
            profile.history, k=k, exclude=profile.owned, disliked=profile.disliked
        )
        games = crud.get_games_by_ids(db, [r.game_id for r in recs])
        items = [
            RecommendedGame(game_id=r.game_id, name=games[r.game_id].name, score=r.score)
            for r in recs
            if r.game_id in games
        ]
        return RecommendationsOut(
            user_id=user_id,
            strategy=self.strategy,
            model_version=self.version,
            recommendations=items,
        )
