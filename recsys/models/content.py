"""M3: recomendador baseado em conteúdo (TF-IDF nos títulos e nas tags).

O dataset só tem o nome dos jogos, então o conteúdo combina três representações:

- **palavras** (e bigramas) do título: captura franquias ("Fallout", "Total War");
- **n-gramas de caracteres** do título: tolera variações ("Half-Life" × "Half Life 2");
- **tags** opcionais, que jogos cadastrados pela API podem trazer. Usam *hashing*
  (sem vocabulário fixo), então uma tag nunca vista também vale na hora.

Os blocos são concatenados e cada jogo vira um vetor de norma 1, então o produto
interno é a similaridade de cosseno. Jogos novos são vetorizados na hora (``add_game``),
sem retreino, e já podem ser recomendados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.base import GAME_COL, NAME_COL, TAGS_COL, BaseRecommender, Recommendation
from recsys.models._common import (
    blocked_indices,
    explained_recommendations,
    game_names,
    history_arrays,
    top_candidates,
)


def _tags_analyzer(tags: list[str]) -> list[str]:
    """As tags já são os termos (função de módulo para o vetorizador ser serializável)."""
    return [t.strip().lower() for t in tags if t and t.strip()]


def _tags_of(games: pd.DataFrame) -> list[list[str]]:
    if TAGS_COL not in games:
        return [[] for _ in range(len(games))]
    return [list(t) if isinstance(t, list | tuple) else [] for t in games[TAGS_COL]]


class ContentRecommender(BaseRecommender):
    """Similaridade de conteúdo entre jogos.

    Args:
        word_weight, char_weight, tag_weight: peso de cada bloco antes da normalização.
    """

    def __init__(
        self, word_weight: float = 1.0, char_weight: float = 1.0, tag_weight: float = 1.0
    ) -> None:
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.tag_weight = tag_weight
        self.word_vectorizer_ = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        self.char_vectorizer_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.tag_vectorizer_ = HashingVectorizer(
            analyzer=_tags_analyzer, n_features=2**12, alternate_sign=False
        )
        self.vectors_: sparse.csr_matrix = sparse.csr_matrix((0, 0))
        self.game_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.game_index_: dict[int, int] = {}
        self.names_: dict[int, str] = {}

    def fit(self, interactions: pd.DataFrame | None, games: pd.DataFrame) -> None:
        """Ajusta os vetorizadores no catálogo. ``interactions`` não é usado."""
        if games is None or games.empty:
            raise ValueError("ContentRecommender precisa do catálogo de jogos")
        games = games.sort_values(GAME_COL)
        names = games[NAME_COL].astype(str).tolist()
        tags = _tags_of(games)

        self.word_vectorizer_.fit(names)
        self.char_vectorizer_.fit(names)
        self.vectors_ = self._vectorize(names, tags)
        self.game_ids_ = games[GAME_COL].to_numpy(dtype=np.int64)
        self.game_index_ = {int(g): i for i, g in enumerate(self.game_ids_)}
        self.names_ = game_names(games)

    def _vectorize(self, names: list[str], tags: list[list[str]]) -> sparse.csr_matrix:
        """Concatena os três blocos (cada um já com norma 1) e normaliza o resultado."""
        blocks = [
            self.word_weight * self.word_vectorizer_.transform(names),
            self.char_weight * self.char_vectorizer_.transform(names),
            self.tag_weight * self.tag_vectorizer_.transform(tags),
        ]
        return normalize(sparse.hstack(blocks).tocsr())

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]:
        """Jogos mais parecidos com o histórico, ponderados pelo score de cada jogo."""
        idx, weights, history_ids = history_arrays(history, self.game_index_, disliked)
        if idx.size == 0:
            return []

        similarity = (self.vectors_[idx] @ self.vectors_.T).toarray()
        contributions = weights[:, None] * similarity
        blocked = blocked_indices(self.game_index_, self._blocked(exclude, disliked))
        candidates = top_candidates(contributions.sum(axis=0), blocked, k)
        return explained_recommendations(
            contributions,
            candidates,
            self.game_ids_,
            history_ids,
            self.names_,
            template="Parecido com {name}, que você jogou",
        )

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]:
        """Jogos com título/tags mais parecidos."""
        if game_id not in self.game_index_:
            return []
        i = self.game_index_[game_id]
        sims = (self.vectors_ @ self.vectors_[i].T).toarray().ravel()
        candidates = top_candidates(sims, np.array([i]), k)
        reason = f"Parecido com {self.names_.get(game_id, f'o jogo {game_id}')}"
        return [Recommendation(int(self.game_ids_[c]), float(sims[c]), reason) for c in candidates]

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Vetoriza um jogo novo (ou atualiza um existente) usando o vocabulário já ajustado."""
        vector = self._vectorize([name], [tags or []])
        self.names_[game_id] = name
        if game_id in self.game_index_:
            vectors = self.vectors_.tolil()
            vectors[self.game_index_[game_id]] = vector
            self.vectors_ = vectors.tocsr()
            return
        self.game_index_[game_id] = len(self.game_ids_)
        self.game_ids_ = np.append(self.game_ids_, np.int64(game_id))
        self.vectors_ = sparse.vstack([self.vectors_, vector]).tocsr()
