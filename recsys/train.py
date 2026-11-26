"""Treino do modelo híbrido servido pela API.

Etapas:

1. (opcional) busca simples de hiperparâmetros em uma amostra de usuários,
   escolhendo pela NDCG@k;
2. avaliação da melhor configuração no protocolo de ``recsys.evaluation``;
3. novo treino com **todos** os dados e gravação do artefato (joblib) em ``artifacts/``,
   com escrita atômica (arquivo temporário + ``os.replace``);
4. registro do treino na tabela ``model_runs`` (parâmetros, métricas e caminho).

Uso:
    python -m recsys.train                    # lê do banco (DATABASE_URL) e registra o treino
    python -m recsys.train --source files     # lê de data/raw/steam, sem registrar
    python -m recsys.train --no-search        # usa os parâmetros padrão
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.base import SCORE_COL
from recsys.data import DEFAULT_DATA_DIR, confidence_score, load_dataset
from recsys.evaluation import DEFAULT_SEED, evaluate, train_test_split
from recsys.models.hybrid import DEFAULT_WEIGHTS, HybridRecommender
from recsys.models.item_knn import ItemKNNRecommender

ARTIFACT_NAME = "model.joblib"
# Valores ajustados na avaliação offline (python -m recsys.tuning; ver docs/modelo.md).
DEFAULT_PARAMS: dict[str, Any] = {
    "alpha": 0.25,
    "k_neighbors": 20,
    "weights": [[size, list(w)] for size, w in DEFAULT_WEIGHTS],
    "dislike_penalty": 0.5,
}
DEFAULT_GRID: dict[str, list[Any]] = {"k_neighbors": [20, 50, 100]}


@dataclass
class TrainingResult:
    """Resultado de um treino completo."""

    model: HybridRecommender
    params: dict[str, Any]
    metrics: dict[str, float]
    artifact_path: str
    trained_at: datetime
    search: list[dict[str, Any]] = field(default_factory=list)


def with_scores(interactions: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Recalcula o score de confiança a partir das horas para o ``alpha`` dado."""
    df = interactions[["user_id", "game_id", "hours"]].copy()
    df[SCORE_COL] = confidence_score(df["hours"].to_numpy(dtype=float), alpha)
    return df


def build_hybrid(params: dict[str, Any]) -> HybridRecommender:
    """Instancia o híbrido a partir de um dicionário de parâmetros (ver ``DEFAULT_PARAMS``)."""
    return HybridRecommender(
        cf=ItemKNNRecommender(k_neighbors=int(params["k_neighbors"])),
        weights=[(int(size), tuple(w)) for size, w in params["weights"]],
        dislike_penalty=float(params["dislike_penalty"]),
    )


def _evaluate_params(
    params: dict[str, Any],
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    k: int,
    max_users: int | None,
    seed: int,
    min_interactions: int,
) -> dict[str, float]:
    scored = with_scores(interactions, params["alpha"])
    split = train_test_split(scored, min_interactions=min_interactions, seed=seed)
    model = build_hybrid(params)
    model.fit(split.train, games)
    return evaluate(model, split, n_items=len(games), k=k, max_users=max_users, seed=seed)


def search_hyperparameters(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    grid: dict[str, list[Any]] | None = None,
    base_params: dict[str, Any] | None = None,
    k: int = 10,
    max_users: int | None = 1000,
    seed: int = DEFAULT_SEED,
    min_interactions: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Testa todas as combinações de ``grid`` e devolve a de maior NDCG@k e a tabela."""
    grid = grid or DEFAULT_GRID
    base = {**DEFAULT_PARAMS, **(base_params or {})}
    names = list(grid)
    results = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = {**base, **dict(zip(names, values, strict=True))}
        metrics = _evaluate_params(
            params, interactions, games, k, max_users, seed, min_interactions
        )
        results.append({**dict(zip(names, values, strict=True)), f"ndcg@{k}": metrics[f"ndcg@{k}"]})
    best = max(results, key=lambda row: row[f"ndcg@{k}"])
    return {**base, **{n: best[n] for n in names}}, results


def save_atomically(model: HybridRecommender, path: Path) -> None:
    """Grava em um arquivo temporário e troca de uma vez (a API nunca lê um arquivo pela metade)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save(str(tmp))
    os.replace(tmp, path)


def train(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    params: dict[str, Any] | None = None,
    search: bool = True,
    grid: dict[str, list[Any]] | None = None,
    k: int = 10,
    search_users: int | None = 1000,
    eval_users: int | None = None,
    seed: int = DEFAULT_SEED,
    min_interactions: int = 5,
) -> TrainingResult:
    """Treina o híbrido, avalia, retreina com tudo e grava o artefato.

    Args:
        interactions: ``user_id``, ``game_id``, ``hours`` (o score é calculado aqui).
        games: catálogo (``game_id``, ``name``, ``tags`` opcional).
        search_users / eval_users: amostra de usuários na busca / na avaliação final.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    table: list[dict[str, Any]] = []
    if search:
        params, table = search_hyperparameters(
            interactions, games, grid, params, k, search_users, seed, min_interactions
        )

    metrics = _evaluate_params(params, interactions, games, k, eval_users, seed, min_interactions)

    trained_at = datetime.now(UTC)
    model = build_hybrid(params)
    model.fit(with_scores(interactions, params["alpha"]), games)
    model.metadata_ = {
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
    }
    path = Path(artifacts_dir) / ARTIFACT_NAME
    save_atomically(model, path)
    return TrainingResult(model, params, metrics, str(path), trained_at, table)


# ---- Banco de dados ----


def load_from_db(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interações positivas (sem "não gostei") e catálogo, lidos do banco da API."""
    from sqlalchemy import select

    from app.db.models import Game, Interaction

    with engine.connect() as conn:
        rows = conn.execute(
            select(Interaction.user_id, Interaction.game_id, Interaction.hours).where(
                Interaction.disliked.is_(False)
            )
        ).all()
        game_rows = conn.execute(select(Game.id, Game.name, Game.tags)).all()
    interactions = pd.DataFrame(rows, columns=["user_id", "game_id", "hours"])
    games = pd.DataFrame(game_rows, columns=["game_id", "name", "tags"])
    return interactions, games


def register_run(engine, result: TrainingResult) -> int:
    """Grava o treino em ``model_runs`` e devolve o ID do registro."""
    from sqlalchemy.orm import Session

    from app.db.models import ModelRun

    with Session(engine) as session, session.begin():
        run = ModelRun(
            trained_at=result.trained_at,
            params=result.params,
            metrics=result.metrics,
            artifact_path=result.artifact_path,
        )
        session.add(run)
        session.flush()
        return run.id


# ---- CLI ----


def main(argv: list[str] | None = None, engine=None) -> TrainingResult:
    parser = argparse.ArgumentParser(description="Treina o modelo híbrido.")
    parser.add_argument("--source", choices=["db", "files"], default="db")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "artifacts"))
    parser.add_argument("--no-search", action="store_true", help="usa os parâmetros padrão")
    parser.add_argument("--search-users", type=int, default=1000)
    parser.add_argument("--eval-users", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    if args.source == "db":
        if engine is None:
            from app.db.session import engine as default_engine

            engine = default_engine
        interactions, games = load_from_db(engine)
    else:
        dataset = load_dataset(args.data_dir)
        interactions, games = dataset.interactions, dataset.games

    print(f"Treinando com {len(interactions)} interações e {len(games)} jogos...")
    result = train(
        interactions,
        games,
        artifacts_dir=args.artifacts_dir,
        search=not args.no_search,
        k=args.k,
        search_users=args.search_users,
        eval_users=args.eval_users,
    )
    for row in result.search:
        print("  busca:", row)
    if args.source == "db":
        run_id = register_run(engine, result)
        print(f"Treino registrado em model_runs (id {run_id}).")

    summary = ", ".join(f"{m}={v:.4f}" for m, v in result.metrics.items() if "@" in m)
    print(f"Parâmetros: k_neighbors={result.params['k_neighbors']}, alpha={result.params['alpha']}")
    print(f"Métricas: {summary}")
    print(f"Artefato: {result.artifact_path} ({time.perf_counter() - start:.0f}s)")
    return result


if __name__ == "__main__":
    main()
