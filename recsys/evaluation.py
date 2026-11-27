"""Avaliação offline dos recomendadores.

Protocolo:

1. Usuários com pelo menos ``min_interactions`` jogos (5, por padrão) entram na avaliação;
   para cada um, ``test_fraction`` (20%) dos jogos vão para o teste, sorteados com semente
   fixa. Os demais usuários ficam inteiros no treino.
2. O modelo é treinado só com o treino. Para cada usuário avaliado, recebe o histórico de
   treino e recomenda ``k`` jogos, excluindo os que ele já tem no treino.
3. Métricas @k (média entre usuários): Precision, Recall, NDCG, MAP e Hit Rate; mais a
   cobertura do catálogo e a novidade dos itens recomendados.

Uso: ``python -m recsys.evaluation [--k 10] [--alpha 1.0] [--max-users N]``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.base import GAME_COL, SCORE_COL, USER_COL, Recommender
from recsys.data import DEFAULT_ALPHA, DEFAULT_DATA_DIR, Dataset, load_dataset

DEFAULT_SEED = 42

# ---- Métricas para um usuário ----


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração dos ``k`` recomendados que são relevantes."""
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fração dos relevantes que aparecem entre os ``k`` recomendados."""
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """1 se ao menos um relevante aparece entre os ``k`` recomendados."""
    return float(bool(set(recommended[:k]) & relevant))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """NDCG com relevância binária: acertos no topo valem mais (desconto ``1/log2(pos+1)``)."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, g in enumerate(recommended[:k]) if g in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal)


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Média da precisão nas posições de acerto, dividida por ``min(|relevantes|, k)``."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, game in enumerate(recommended[:k], start=1):
        if game in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


# ---- Métricas do conjunto de recomendações ----


def catalog_coverage(recommendations: Iterable[list[int]], n_items: int) -> float:
    """Fração do catálogo que aparece em alguma lista recomendada."""
    if n_items <= 0:
        return 0.0
    distinct = set().union(*map(set, recommendations))
    return len(distinct) / n_items


def novelty(recommendations: Iterable[list[int]], popularity: Mapping[int, float]) -> float:
    """Autoinformação média ``-log2(p)`` dos itens recomendados (maior = menos óbvio).

    ``popularity`` é a fração de usuários que tem cada jogo no treino; jogos sem nenhum
    usuário usam a menor fração positiva observada.
    """
    floor = min((p for p in popularity.values() if p > 0), default=1.0)
    values = [
        -np.log2(max(popularity.get(g, 0.0), floor)) for recs in recommendations for g in recs
    ]
    return float(np.mean(values)) if values else 0.0


def average_popularity(
    recommendations: Iterable[list[int]], popularity: Mapping[int, float]
) -> float:
    """Popularidade média (fração de usuários) dos itens recomendados."""
    values = [popularity.get(g, 0.0) for recs in recommendations for g in recs]
    return float(np.mean(values)) if values else 0.0


# ---- Divisão treino/teste ----


@dataclass
class Split:
    """Interações de treino e jogos de teste de cada usuário avaliado."""

    train: pd.DataFrame
    test: dict[int, set[int]]


def train_test_split(
    interactions: pd.DataFrame,
    min_interactions: int = 5,
    test_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Separa ``test_fraction`` dos jogos de cada usuário com histórico suficiente.

    Cada usuário avaliado tem ``max(1, round(test_fraction * n))`` jogos no teste.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction deve estar entre 0 e 1")
    df = interactions.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    sizes = df.groupby(USER_COL)[GAME_COL].transform("size")
    n_test = np.maximum(1, np.round(test_fraction * sizes)).astype(int)
    order = pd.Series(rng.random(len(df)), index=df.index)
    rank = order.groupby(df[USER_COL]).rank(method="first") - 1
    is_test = (sizes >= min_interactions) & (rank < n_test)

    test_rows = df[is_test]
    test = {int(u): set(map(int, g)) for u, g in test_rows.groupby(USER_COL)[GAME_COL]}
    return Split(train=df[~is_test].reset_index(drop=True), test=test)


# ---- Avaliação ----


def _histories(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    return {
        int(u): dict(zip(g[GAME_COL].astype(int), g[SCORE_COL].astype(float), strict=True))
        for u, g in train.groupby(USER_COL)
    }


def evaluate(
    model: Recommender,
    split: Split,
    n_items: int,
    k: int = 10,
    max_users: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Avalia um modelo já treinado em ``split.train``.

    Args:
        n_items: tamanho do catálogo (denominador da cobertura).
        max_users: se dado, avalia uma amostra fixa de usuários (para testes rápidos).
    """
    histories = _histories(split.train)
    users = sorted(split.test)
    if max_users is not None and max_users < len(users):
        users = sorted(np.random.default_rng(seed).choice(users, max_users, replace=False))

    n_train_users = split.train[USER_COL].nunique()
    counts = split.train.groupby(GAME_COL)[USER_COL].nunique()
    popularity = (counts / n_train_users).to_dict() if n_train_users else {}

    metrics = {"precision": [], "recall": [], "ndcg": [], "map": [], "hit_rate": []}
    all_recs: list[list[int]] = []
    start = time.perf_counter()
    for user in users:
        history = histories.get(int(user), {})
        recs = [r.game_id for r in model.recommend(history, k=k, exclude=set(history))]
        relevant = split.test[int(user)]
        all_recs.append(recs)
        metrics["precision"].append(precision_at_k(recs, relevant, k))
        metrics["recall"].append(recall_at_k(recs, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant, k))
        metrics["map"].append(average_precision_at_k(recs, relevant, k))
        metrics["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
    elapsed = time.perf_counter() - start

    result = {f"{name}@{k}": float(np.mean(v)) if v else 0.0 for name, v in metrics.items()}
    result["coverage"] = catalog_coverage(all_recs, n_items)
    result["novelty"] = novelty(all_recs, popularity)
    result["avg_popularity"] = average_popularity(all_recs, popularity)
    result["users"] = float(len(users))
    result["ms_per_user"] = 1000 * elapsed / len(users) if users else 0.0
    return result


def compare_models(
    models: Mapping[str, Recommender],
    dataset: Dataset,
    k: int = 10,
    seed: int = DEFAULT_SEED,
    max_users: int | None = None,
) -> pd.DataFrame:
    """Treina e avalia cada modelo no mesmo split; uma linha por modelo."""
    split = train_test_split(dataset.interactions, seed=seed)
    rows = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(split.train, dataset.games)
        fit_seconds = time.perf_counter() - start
        rows[name] = evaluate(
            model, split, n_items=len(dataset.games), k=k, max_users=max_users, seed=seed
        )
        rows[name]["fit_s"] = fit_seconds
    return pd.DataFrame.from_dict(rows, orient="index")


def to_markdown(table: pd.DataFrame, digits: int = 4) -> str:
    """Tabela em Markdown (sem depender do pacote ``tabulate``)."""
    header = "| modelo | " + " | ".join(table.columns) + " |"
    sep = "|---|" + "---:|" * len(table.columns)
    lines = [header, sep]
    for name, row in table.iterrows():
        cells = [f"{v:.0f}" if col == "users" else f"{v:.{digits}f}" for col, v in row.items()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def default_models() -> dict[str, Recommender]:
    """M0 a M3 com os hiperparâmetros iniciais do planejamento."""
    from recsys.models.content import ContentRecommender
    from recsys.models.item_knn import ItemKNNRecommender
    from recsys.models.popularity import PopularityRecommender
    from recsys.models.svd import SVDRecommender

    return {
        "M0 popularidade": PopularityRecommender(),
        "M1 item-kNN (k=50)": ItemKNNRecommender(k_neighbors=50),
        "M2 SVD (64 fatores)": SVDRecommender(n_factors=64),
        "M3 conteúdo": ContentRecommender(),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: compara M0 a M3 com os parâmetros iniciais e imprime a tabela em Markdown."""
    parser = argparse.ArgumentParser(description="Compara os recomendadores (avaliação offline).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data_dir, alpha=args.alpha)
    table = compare_models(
        default_models(), dataset, k=args.k, seed=args.seed, max_users=args.max_users
    )
    print(to_markdown(table))


if __name__ == "__main__":
    main()
