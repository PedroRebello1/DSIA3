"""Ponte entre a API e o pacote ``recsys``.

O modelo recebe só o histórico (``{game_id: score}``); este serviço monta esse histórico
a partir do banco, chama o modelo e devolve a resposta com os nomes dos jogos.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ModelUnavailableError
from app.db import crud
from app.db.models import Game, Interaction
from app.schemas.recommendation import (
    RecommendationsOut,
    RecommendedGame,
    SimilarGamesOut,
    Strategy,
)
from app.services.games import get_game
from app.services.users import require_user
from recsys.base import Recommendation, Recommender
from recsys.data import DEFAULT_ALPHA, confidence_score
from recsys.models.content import ContentRecommender
from recsys.models.hybrid import HybridRecommender
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


def to_items(db: Session, recs: list[Recommendation]) -> list[RecommendedGame]:
    """Acrescenta o nome de cada jogo (ignora IDs que não existem mais no banco)."""
    games = crud.get_games_by_ids(db, [r.game_id for r in recs])
    return [
        RecommendedGame(
            game_id=r.game_id, name=games[r.game_id].name, score=r.score, reason=r.reason
        )
        for r in recs
        if r.game_id in games
    ]


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
    """Guarda o modelo em memória e atende os pedidos de recomendação.

    ``similar_model`` responde ``/games/{id}/similar``; por padrão é o próprio ``model``.
    """

    def __init__(
        self,
        model: Recommender,
        strategy: str,
        version: str,
        alpha: float = DEFAULT_ALPHA,
        similar_model: Recommender | None = None,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.version = version
        self.alpha = alpha
        self.similar_model = similar_model if similar_model is not None else model
        self._lock = threading.Lock()

    @property
    def strategies(self) -> dict[Strategy, Recommender]:
        """Modelos disponíveis por estratégia (o híbrido expõe seus componentes)."""
        if isinstance(self.model, HybridRecommender):
            return {
                Strategy.hybrid: self.model,
                Strategy.cf: self.model.cf,
                Strategy.content: self.model.content,
                Strategy.popular: self.model.popularity,
            }
        return {Strategy(self.strategy): self.model}

    def model_for(self, strategy: Strategy | None) -> tuple[Strategy, Recommender]:
        """Modelo da estratégia pedida (ou da padrão); 503 se ela não estiver disponível."""
        chosen = Strategy(strategy or self.strategy)
        available = self.strategies
        if chosen not in available:
            names = ", ".join(sorted(available))
            raise ModelUnavailableError(
                f"Estratégia '{chosen}' indisponível: o modelo híbrido não está carregado "
                f"(disponíveis: {names}). Treine o modelo com POST /model/retrain."
            )
        return chosen, available[chosen]

    @classmethod
    def from_db(cls, db: Session, alpha: float = DEFAULT_ALPHA) -> RecommenderService:
        """Treina, com os dados atuais do banco, popularidade (recomendações) e conteúdo
        (jogos similares)."""
        interactions, games = load_training_data(db, alpha)
        model = PopularityRecommender()
        model.fit(interactions, games)
        similar = None
        if not games.empty:
            similar = ContentRecommender()
            similar.fit(None, games)
        version = datetime.now(UTC).isoformat(timespec="seconds")
        return cls(model, "popular", version, alpha=alpha, similar_model=similar)

    @classmethod
    def from_artifact(cls, path: str | Path) -> RecommenderService:
        """Carrega o híbrido gravado por ``recsys.train`` (versão = data do treino)."""
        model = HybridRecommender.load(str(path))
        metadata = getattr(model, "metadata_", {}) or {}
        version = metadata.get("trained_at") or datetime.now(UTC).isoformat(timespec="seconds")
        alpha = float(metadata.get("params", {}).get("alpha", DEFAULT_ALPHA))
        return cls(model, "hybrid", version, alpha=alpha)

    def recommend(
        self,
        db: Session,
        user_id: int,
        k: int,
        strategy: Strategy | None = None,
        exclude_owned: bool = True,
    ) -> RecommendationsOut:
        """Recomenda ``k`` jogos para o usuário com a estratégia pedida.

        Jogos marcados como "não gostei" nunca voltam; os demais jogos do histórico só são
        excluídos se ``exclude_owned`` for verdadeiro.
        """
        require_user(db, user_id)
        chosen, model = self.model_for(strategy)
        profile = build_profile(crud.get_interactions(db, user_id), self.alpha)
        exclude = profile.owned if exclude_owned else set()
        recs = model.recommend(profile.history, k=k, exclude=exclude, disliked=profile.disliked)
        return RecommendationsOut(
            user_id=user_id,
            strategy=chosen,
            model_version=self.version,
            recommendations=to_items(db, recs),
        )

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Avisa os modelos sobre um jogo novo (o conteúdo já pode recomendá-lo)."""
        with self._lock:
            for model in {id(m): m for m in (self.model, self.similar_model)}.values():
                model.add_game(game_id, name, tags)

    def similar_games(self, db: Session, game_id: int, k: int) -> SimilarGamesOut:
        """Jogos parecidos com ``game_id`` (404 se o jogo não existir)."""
        game = get_game(db, game_id)
        recs = self.similar_model.similar_games(game_id, k=k)
        return SimilarGamesOut(game_id=game.id, name=game.name, similar=to_items(db, recs))
