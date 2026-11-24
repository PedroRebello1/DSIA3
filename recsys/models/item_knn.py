"""M1: item-kNN com similaridade de cosseno entre jogos.

Cada jogo é representado pela sua coluna na matriz ``usuários × jogos`` (score de
confiança). Guardamos só os ``k_neighbors`` vizinhos mais parecidos de cada jogo, em uma
matriz esparsa, o que mantém o modelo pequeno (5 mil jogos × 50 vizinhos).

O score de um candidato ``j`` para um histórico ``h`` é ``Σ_i h_i · sim(i, j)``, somando
apenas os jogos ``i`` do histórico que têm ``j`` entre seus vizinhos. Como o modelo só
precisa do histórico, preferências novas valem na hora, sem retreino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.base import GAME_COL, BaseRecommender, Recommendation
from recsys.data import build_matrix
from recsys.models._common import (
    blocked_indices,
    explained_recommendations,
    game_names,
    history_arrays,
    top_candidates,
)


class ItemKNNRecommender(BaseRecommender):
    """Recomendador colaborativo item a item.

    Args:
        k_neighbors: quantos vizinhos guardar por jogo.
        chunk_size: jogos processados por vez no cálculo da similaridade (controla memória).
    """

    def __init__(self, k_neighbors: int = 50, chunk_size: int = 1024) -> None:
        if k_neighbors < 1:
            raise ValueError("k_neighbors deve ser >= 1")
        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
        self.similarity_: sparse.csr_matrix = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.game_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.game_index_: dict[int, int] = {}
        self.names_: dict[int, str] = {}

    def fit(self, interactions: pd.DataFrame, games: pd.DataFrame | None = None) -> None:
        """Calcula a similaridade de cosseno e mantém os top-k vizinhos de cada jogo."""
        game_ids = np.unique(interactions[GAME_COL].to_numpy(dtype=np.int64))
        if games is not None:
            game_ids = np.union1d(game_ids, games[GAME_COL].to_numpy(dtype=np.int64))
            self.names_ = game_names(games)

        matrix = build_matrix(interactions, game_ids=game_ids).matrix
        self.game_ids_ = game_ids
        self.game_index_ = {int(g): i for i, g in enumerate(game_ids)}
        self.similarity_ = self._top_k_cosine(matrix)

    def _top_k_cosine(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        n_items = matrix.shape[1]
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        normalized = (matrix @ sparse.diags(inv)).tocsc()
        items_by_users = normalized.T.tocsr()

        kk = min(self.k_neighbors, max(n_items - 1, 0))
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.chunk_size):
            end = min(start + self.chunk_size, n_items)
            block = (items_by_users[start:end] @ normalized).toarray()
            block[np.arange(end - start), np.arange(start, end)] = 0.0  # sem auto-similaridade
            if kk == 0:
                continue
            top = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
            top_vals = np.take_along_axis(block, top, axis=1)
            keep = top_vals > 0
            rows.append(np.repeat(np.arange(start, end), kk)[keep.ravel()])
            cols.append(top[keep])
            vals.append(top_vals[keep])

        if not rows:
            return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
            dtype=np.float32,
        )

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]:
        """Recomenda vizinhos dos jogos do histórico; histórico vazio devolve lista vazia."""
        idx, weights, history_ids = history_arrays(history, self.game_index_, disliked)
        if idx.size == 0:
            return []

        contributions = self.similarity_[idx].multiply(weights[:, None]).toarray()
        blocked = blocked_indices(self.game_index_, self._blocked(exclude, disliked))
        candidates = top_candidates(contributions.sum(axis=0), blocked, k)
        return explained_recommendations(
            contributions, candidates, self.game_ids_, history_ids, self.names_
        )

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]:
        """Os ``k`` vizinhos mais parecidos do jogo (lista vazia se o jogo for desconhecido)."""
        if game_id not in self.game_index_:
            return []
        row = self.similarity_[self.game_index_[game_id]].toarray().ravel()
        candidates = top_candidates(row, np.array([], dtype=np.int64), k)
        name = self.names_.get(game_id, f"o jogo {game_id}")
        reason = f"Quem jogou {name} também jogou"
        return [Recommendation(int(self.game_ids_[c]), float(row[c]), reason) for c in candidates]

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Registra um jogo novo, ainda sem vizinhos (não há interações com ele)."""
        self.names_[game_id] = name
        if game_id in self.game_index_:
            return
        n = len(self.game_ids_)
        self.game_index_[game_id] = n
        self.game_ids_ = np.append(self.game_ids_, np.int64(game_id))
        sim = self.similarity_.tocoo()
        self.similarity_ = sparse.csr_matrix(
            (sim.data, (sim.row, sim.col)), shape=(n + 1, n + 1), dtype=np.float32
        )
