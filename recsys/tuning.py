"""Ajuste dos pesos do híbrido e do α do score de confiança.

Experimentos (todos no mesmo split 80/20, semente fixa):

1. **α** do score ``1 + α·log1p(horas)``: 0,5 / 1 / 2;
2. **pesos da faixa "≥ 3 jogos"**: avaliados no protocolo padrão (usuários com ≥ 5 jogos);
3. **pesos da faixa "1–2 jogos"**: o modelo é treinado com todos os dados de treino, mas
   recebe só 2 jogos do histórico de cada usuário avaliado (simula um usuário quase novo).

Uso: ``python -m recsys.tuning [--max-users 1500]`` (imprime as tabelas em Markdown).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import pandas as pd

from recsys.base import USER_COL
from recsys.data import load_dataset
from recsys.evaluation import DEFAULT_SEED, Split, evaluate, to_markdown, train_test_split
from recsys.train import DEFAULT_PARAMS, build_hybrid, with_scores

Weights = tuple[float, float, float]

ALPHAS = [0.5, 1.0, 2.0]
WEIGHTS_LONG: list[Weights] = [
    (0.7, 0.2, 0.1),
    (0.8, 0.1, 0.1),
    (0.6, 0.3, 0.1),
    (0.8, 0.2, 0.0),
    (0.9, 0.1, 0.0),
    (1.0, 0.0, 0.0),
]
WEIGHTS_SHORT: list[Weights] = [
    (0.4, 0.4, 0.2),
    (0.5, 0.3, 0.2),
    (0.6, 0.2, 0.2),
    (0.7, 0.2, 0.1),
    (0.3, 0.3, 0.4),
    (1.0, 0.0, 0.0),
]
REPORTED = ["precision@10", "recall@10", "ndcg@10", "map@10", "hit_rate@10", "coverage", "novelty"]


def with_band(params: dict[str, Any], min_size: int, weights: Weights) -> dict[str, Any]:
    """Copia ``params`` trocando os pesos da faixa que começa em ``min_size``."""
    bands = {int(size): list(w) for size, w in params["weights"]}
    bands[min_size] = list(weights)
    return {**params, "weights": [[size, bands[size]] for size in sorted(bands)]}


def truncate_histories(split: Split, limit: int, seed: int = DEFAULT_SEED) -> Split:
    """Mantém no treino só ``limit`` jogos de cada usuário avaliado (os demais ficam inteiros)."""
    evaluated = split.train[USER_COL].isin(split.test)
    sampled = (
        split.train[evaluated]
        .groupby(USER_COL, group_keys=False)
        .sample(frac=1.0, random_state=seed)
        .groupby(USER_COL)
        .head(limit)
    )
    return Split(train=pd.concat([split.train[~evaluated], sampled]), test=split.test)


def run_experiment(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    params: dict[str, Any],
    history_limit: int | None = None,
    max_users: int | None = None,
    seed: int = DEFAULT_SEED,
    min_interactions: int = 5,
) -> dict[str, float]:
    """Treina o híbrido com ``params`` e avalia (opcionalmente com histórico truncado)."""
    split = train_test_split(
        with_scores(interactions, params["alpha"]), min_interactions=min_interactions, seed=seed
    )
    model = build_hybrid(params)
    model.fit(split.train, games)
    if history_limit is not None:
        split = truncate_histories(split, history_limit, seed)
    return evaluate(model, split, n_items=len(games), max_users=max_users, seed=seed)


def _table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")[REPORTED]


def _label(weights: Sequence[float]) -> str:
    return " / ".join(f"{w:g}" for w in weights)


def tune(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    base_params: dict[str, Any] | None = None,
    max_users: int | None = 1500,
    seed: int = DEFAULT_SEED,
    alphas: Sequence[float] = ALPHAS,
    weights_long: Sequence[Weights] = WEIGHTS_LONG,
    weights_short: Sequence[Weights] = WEIGHTS_SHORT,
    min_interactions: int = 5,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Roda os três experimentos em sequência, cada um partindo do melhor anterior (NDCG@10)."""
    params = {**DEFAULT_PARAMS, **(base_params or {})}
    common = {"max_users": max_users, "seed": seed, "min_interactions": min_interactions}
    tables = {}

    rows = {
        f"α = {a:g}": run_experiment(interactions, games, {**params, "alpha": a}, **common)
        for a in alphas
    }
    tables["alpha"] = _table(rows)
    params["alpha"] = list(alphas)[int(tables["alpha"]["ndcg@10"].to_numpy().argmax())]

    rows = {
        _label(w): run_experiment(interactions, games, with_band(params, 3, w), **common)
        for w in weights_long
    }
    tables["weights_long"] = _table(rows)
    best_long = list(weights_long)[int(tables["weights_long"]["ndcg@10"].to_numpy().argmax())]
    params = with_band(params, 3, best_long)

    rows = {
        _label(w): run_experiment(
            interactions, games, with_band(params, 1, w), history_limit=2, **common
        )
        for w in weights_short
    }
    tables["weights_short"] = _table(rows)
    best_short = list(weights_short)[int(tables["weights_short"]["ndcg@10"].to_numpy().argmax())]
    params = with_band(params, 1, best_short)
    return params, tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta α e pesos do híbrido.")
    parser.add_argument("--max-users", type=int, default=1500)
    parser.add_argument("--k-neighbors", type=int, default=20)
    args = parser.parse_args(argv)

    dataset = load_dataset()
    params, tables = tune(
        dataset.interactions,
        dataset.games,
        base_params={"k_neighbors": args.k_neighbors},
        max_users=args.max_users,
    )
    titles = {
        "alpha": "α do score de confiança",
        "weights_long": "Pesos CF / conteúdo / popularidade, histórico ≥ 3",
        "weights_short": "Pesos CF / conteúdo / popularidade, histórico de 2 jogos",
    }
    for key, table in tables.items():
        print(f"\n### {titles[key]}\n")
        print(to_markdown(table))
    print(f"\nParâmetros escolhidos: {params}")


if __name__ == "__main__":
    main()
