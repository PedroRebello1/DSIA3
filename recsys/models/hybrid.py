"""M4: recomendador híbrido (modelo servido pela API).

Combina três componentes, todos acessados só pelo contrato ``Recommender``:

- **CF** (item-kNN, por padrão): melhor acurácia para quem tem histórico;
- **conteúdo** (TF-IDF de títulos/tags): jogos novos e franquias;
- **popularidade**: fallback de cold start.

Para cada pedido, cada componente devolve um conjunto de candidatos; os scores são
normalizados (min-max) por componente e somados com pesos que dependem do tamanho do
histórico do usuário. Jogos parecidos com os marcados como "não gostei" são penalizados.
O ``reason`` vem do componente que mais contribuiu para cada jogo.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from recsys.base import BaseRecommender, Recommendation, Recommender
from recsys.models.content import ContentRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender

Weights = tuple[float, float, float]  # (CF, conteúdo, popularidade)

# (tamanho mínimo do histórico, pesos): tabela da seção 4.2 do planejamento.
DEFAULT_WEIGHTS: tuple[tuple[int, Weights], ...] = (
    (0, (0.0, 0.0, 1.0)),
    (1, (0.4, 0.4, 0.2)),
    (3, (0.7, 0.2, 0.1)),
)
COMPONENTS = ("cf", "content", "popularity")


def normalize_scores(recs: Sequence[Recommendation]) -> dict[int, float]:
    """Min-max dos scores para [0, 1]; se todos forem iguais, todos valem 1."""
    if not recs:
        return {}
    scores = [r.score for r in recs]
    low, high = min(scores), max(scores)
    if high == low:
        return {r.game_id: 1.0 for r in recs}
    return {r.game_id: (r.score - low) / (high - low) for r in recs}


class HybridRecommender(BaseRecommender):
    """Combinação ponderada de CF, conteúdo e popularidade.

    Args:
        cf, content, popularity: componentes (padrão: item-kNN, TF-IDF e popularidade).
        weights: pares ``(histórico mínimo, (peso CF, peso conteúdo, peso popularidade))``.
        dislike_penalty: quanto subtrair (escala 0–1) de jogos parecidos com os "não gostei".
        candidate_pool: candidatos pedidos a cada componente (mínimo; cresce com ``k``).
    """

    def __init__(
        self,
        cf: Recommender | None = None,
        content: Recommender | None = None,
        popularity: Recommender | None = None,
        weights: Sequence[tuple[int, Weights]] = DEFAULT_WEIGHTS,
        dislike_penalty: float = 0.5,
        candidate_pool: int = 100,
    ) -> None:
        if not weights or min(size for size, _ in weights) != 0:
            raise ValueError("weights precisa de uma faixa começando em histórico 0")
        if dislike_penalty < 0:
            raise ValueError("dislike_penalty deve ser >= 0")
        self.cf = cf if cf is not None else ItemKNNRecommender()
        self.content = content if content is not None else ContentRecommender()
        self.popularity = popularity if popularity is not None else PopularityRecommender()
        self.weights = tuple(sorted((int(s), tuple(map(float, w))) for s, w in weights))
        self.dislike_penalty = dislike_penalty
        self.candidate_pool = candidate_pool
        self.metadata_: dict = {}

    def fit(self, interactions: pd.DataFrame, games: pd.DataFrame) -> None:
        """Treina os três componentes com os mesmos dados."""
        for name in COMPONENTS:
            getattr(self, name).fit(interactions, games)

    def weights_for(self, history_size: int) -> Weights:
        """Pesos da maior faixa de histórico que o usuário atinge."""
        chosen = self.weights[0][1]
        for min_size, weights in self.weights:
            if history_size >= min_size:
                chosen = weights
        return chosen

    def _dislike_penalties(self, disliked: set[int], blocked: set[int], pool: int) -> dict:
        """Similaridade (0–1) de cada candidato com os jogos marcados como "não gostei"."""
        if not disliked or self.dislike_penalty == 0:
            return {}
        history = dict.fromkeys(disliked, 1.0)
        penalties: dict[int, float] = {}
        for model in (self.cf, self.content):
            similar = normalize_scores(model.recommend(history, k=pool, exclude=blocked))
            for game_id, value in similar.items():
                penalties[game_id] = max(penalties.get(game_id, 0.0), value)
        return penalties

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]:
        """Combina os componentes com pesos pelo tamanho do histórico positivo."""
        disliked = disliked or set()
        positive = {g: s for g, s in history.items() if g not in disliked and s > 0}
        blocked = self._blocked(exclude, disliked)
        pool = max(self.candidate_pool, 5 * k)

        combined: dict[int, float] = {}
        best: dict[int, tuple[float, str | None]] = {}
        for name, weight in zip(COMPONENTS, self.weights_for(len(positive)), strict=True):
            if weight == 0:
                continue
            recs = getattr(self, name).recommend(positive, k=pool, exclude=blocked)
            normalized = normalize_scores(recs)
            for rec in recs:
                part = weight * normalized[rec.game_id]
                combined[rec.game_id] = combined.get(rec.game_id, 0.0) + part
                if part > best.get(rec.game_id, (-1.0, None))[0]:
                    best[rec.game_id] = (part, rec.reason)

        for game_id, penalty in self._dislike_penalties(disliked, blocked, pool).items():
            if game_id in combined:
                combined[game_id] -= self.dislike_penalty * penalty

        ranked = sorted(
            (item for item in combined.items() if item[1] > 0), key=lambda x: (-x[1], x[0])
        )
        return [Recommendation(g, s, best[g][1]) for g, s in ranked[:k]]

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]:
        """Média dos jogos parecidos segundo CF (padrão de jogo) e conteúdo (título/tags)."""
        combined: dict[int, float] = {}
        best: dict[int, tuple[float, str | None]] = {}
        for model in (self.cf, self.content):
            recs = model.similar_games(game_id, k=max(self.candidate_pool, 5 * k))
            normalized = normalize_scores(recs)
            for rec in recs:
                part = 0.5 * normalized[rec.game_id]
                combined[rec.game_id] = combined.get(rec.game_id, 0.0) + part
                if part > best.get(rec.game_id, (-1.0, None))[0]:
                    best[rec.game_id] = (part, rec.reason)
        ranked = sorted(combined.items(), key=lambda x: (-x[1], x[0]))
        return [Recommendation(g, s, best[g][1]) for g, s in ranked[:k]]

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None:
        """Repassa o jogo novo a todos os componentes (o conteúdo já o torna recomendável)."""
        for name_ in COMPONENTS:
            getattr(self, name_).add_game(game_id, name, tags)
