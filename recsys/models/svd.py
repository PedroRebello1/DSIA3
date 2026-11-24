"""M2: fatoração de matrizes com ``TruncatedSVD``.

A matriz ``usuários × jogos`` (score de confiança) é aproximada por ``U Σ Vᵀ``. Cada jogo
vira um vetor latente (linha de ``V``). Para um histórico ``h`` qualquer, o usuário é
projetado no espaço latente (*fold-in*: ``u = h V``) e os scores são ``u Vᵀ``. Assim
usuários novos ou atualizados não precisam de retreino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recsys.base import GAME_COL, BaseRecommender, Recommendation
from recsys.data import build_matrix
from recsys.models._common import (
    blocked_indices,
    explained_recommendations,
    game_names,
    history_arrays,
    top_candidates,
)


class SVDRecommender(BaseRecommender):
    """Recomendador por fatores latentes.

    Args:
        n_factors: dimensão do espaço latente (limitada pelo tamanho da matriz).
        random_state: semente do SVD randomizado, para resultados reprodutíveis.
    """

    def __init__(self, n_factors: int = 64, random_state: int = 42) -> None:
        if n_factors < 1:
            raise ValueError("n_factors deve ser >= 1")
        self.n_factors = n_factors
        self.random_state = random_state
        self.item_factors_: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.explained_variance_ratio_: float = 0.0
        self.game_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.game_index_: dict[int, int] = {}
        self.names_: dict[int, str] = {}

    def fit(self, interactions: pd.DataFrame, games: pd.DataFrame | None = None) -> None:
        """Fatora a matriz de confiança e guarda os fatores dos jogos."""
        game_ids = np.unique(interactions[GAME_COL].to_numpy(dtype=np.int64))
        if games is not None:
            game_ids = np.union1d(game_ids, games[GAME_COL].to_numpy(dtype=np.int64))
            self.names_ = game_names(games)

        matrix = build_matrix(interactions, game_ids=game_ids).matrix
        self.game_ids_ = game_ids
        self.game_index_ = {int(g): i for i, g in enumerate(game_ids)}

        n_components = min(self.n_factors, min(matrix.shape) - 1)
        if n_components < 1:
            self.item_factors_ = np.zeros((len(game_ids), 1), dtype=np.float32)
            self.explained_variance_ratio_ = 0.0
            return
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        svd.fit(matrix)
        self.item_factors_ = svd.components_.T.astype(np.float32)
        self.explained_variance_ratio_ = float(svd.explained_variance_ratio_.sum())

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]:
        """Pontua todos os jogos por ``h V Vᵀ``; histórico vazio devolve lista vazia."""
        idx, weights, history_ids = history_arrays(history, self.game_index_, disliked)
        if idx.size == 0:
            return []

        factors = self.item_factors_
        contributions = weights[:, None] * (factors[idx] @ factors.T)
        blocked = blocked_indices(self.game_index_, self._blocked(exclude, disliked))
        candidates = top_candidates(contributions.sum(axis=0), blocked, k)
        return explained_recommendations(
            contributions, candidates, self.game_ids_, history_ids, self.names_
        )

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]:
        """Jogos com vetor latente mais próximo (cosseno)."""
        if game_id not in self.game_index_:
            return []
        norms = np.linalg.norm(self.item_factors_, axis=1)
        normalized = self.item_factors_ / np.where(norms > 0, norms, 1.0)[:, None]
        i = self.game_index_[game_id]
        sims = normalized @ normalized[i]
        candidates = top_candidates(sims, np.array([i]), k)
        name = self.names_.get(game_id, f"o jogo {game_id}")
        reason = f"Quem jogou {name} também jogou"
        return [Recommendation(int(self.game_ids_[c]), float(sims[c]), reason) for c in candidates]

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Registra um jogo novo com vetor latente nulo (sem interações, sem sinal)."""
        self.names_[game_id] = name
        if game_id in self.game_index_:
            return
        self.game_index_[game_id] = len(self.game_ids_)
        self.game_ids_ = np.append(self.game_ids_, np.int64(game_id))
        zeros = np.zeros((1, self.item_factors_.shape[1]), dtype=np.float32)
        self.item_factors_ = np.vstack([self.item_factors_, zeros])
