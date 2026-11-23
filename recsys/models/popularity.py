"""M0: recomendador por popularidade (baseline e fallback de cold start)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from recsys.base import GAME_COL, SCORE_COL, USER_COL, BaseRecommender, Recommendation

POPULAR_REASON = "Popular entre jogadores"


class PopularityRecommender(BaseRecommender):
    """Ordena os jogos pela popularidade no conjunto de treino.

    Args:
        by: ``"users"`` conta quantos usuários distintos têm o jogo;
            ``"score"`` soma o score de confiança (favorece jogos muito jogados).
    """

    def __init__(self, by: Literal["users", "score"] = "users") -> None:
        if by not in ("users", "score"):
            raise ValueError("by deve ser 'users' ou 'score'")
        self.by = by
        self.popularity_: dict[int, float] = {}
        self._ranking: list[int] = []

    def fit(self, interactions: pd.DataFrame, games: pd.DataFrame | None = None) -> None:
        """Calcula a popularidade de cada jogo, normalizada para o intervalo [0, 1]."""
        if self.by == "users":
            counts = interactions.groupby(GAME_COL)[USER_COL].nunique()
        else:
            counts = interactions.groupby(GAME_COL)[SCORE_COL].sum()

        top = counts.max() if len(counts) else 0.0
        popularity = (counts / top) if top > 0 else counts.astype(float)
        self.popularity_ = {int(g): float(p) for g, p in popularity.items()}

        if games is not None:
            for game_id in games[GAME_COL]:
                self.popularity_.setdefault(int(game_id), 0.0)
        self._rebuild_ranking()

    def _rebuild_ranking(self) -> None:
        # Desempate pelo game_id para o resultado ser determinístico.
        self._ranking = sorted(self.popularity_, key=lambda g: (-self.popularity_[g], g))

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]:
        """Retorna os ``k`` jogos mais populares que não estão bloqueados.

        O histórico não altera a ordem; ele existe só para seguir o contrato.
        """
        blocked = self._blocked(exclude, disliked)
        recs: list[Recommendation] = []
        for game_id in self._ranking:
            if len(recs) >= k:
                break
            if game_id in blocked:
                continue
            recs.append(Recommendation(game_id, self.popularity_[game_id], POPULAR_REASON))
        return recs

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]:
        """Sem noção de similaridade: devolve os mais populares, exceto o próprio jogo."""
        return self.recommend({}, k=k, exclude={game_id})

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Registra um jogo novo com popularidade zero (vai para o fim do ranking)."""
        if game_id not in self.popularity_:
            self.popularity_[game_id] = 0.0
            self._ranking.append(game_id)
