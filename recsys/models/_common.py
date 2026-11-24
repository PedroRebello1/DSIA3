"""Funções compartilhadas pelos recomendadores baseados em similaridade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.base import GAME_COL, NAME_COL, Recommendation


def game_names(games: pd.DataFrame) -> dict[int, str]:
    """Mapa ``game_id -> nome`` a partir do catálogo."""
    return {int(g): str(n) for g, n in zip(games[GAME_COL], games[NAME_COL], strict=True)}


def history_arrays(
    history: dict[int, float],
    game_index: dict[int, int],
    disliked: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Converte o histórico em índices e pesos, ignorando jogos desconhecidos e "não gostei".

    Returns:
        ``(indices, pesos, game_ids)`` na mesma ordem.
    """
    disliked = disliked or set()
    pairs = [
        (game_index[g], float(w), g)
        for g, w in history.items()
        if g in game_index and g not in disliked and w > 0
    ]
    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), []
    idx, weights, ids = zip(*pairs, strict=True)
    return np.asarray(idx, dtype=np.int64), np.asarray(weights, dtype=np.float64), list(ids)


def blocked_indices(game_index: dict[int, int], blocked: set[int]) -> np.ndarray:
    """Índices (na matriz do modelo) dos jogos que não podem ser recomendados."""
    return np.asarray([game_index[g] for g in blocked if g in game_index], dtype=np.int64)


def top_candidates(scores: np.ndarray, blocked: np.ndarray, k: int) -> np.ndarray:
    """Índices dos ``k`` maiores scores positivos, fora dos bloqueados.

    Empates são resolvidos pelo índice (ordem do ``game_id``), para resultados determinísticos.
    """
    if k <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[blocked] = -np.inf
    valid = np.flatnonzero(scores > 0)
    order = np.lexsort((valid, -scores[valid]))
    return valid[order[:k]]


def explained_recommendations(
    contributions: np.ndarray,
    candidates: np.ndarray,
    game_ids: np.ndarray,
    history_ids: list[int],
    names: dict[int, str],
    template: str = "Porque você jogou {name}",
) -> list[Recommendation]:
    """Monta as recomendações com o jogo do histórico que mais contribuiu para cada uma.

    Args:
        contributions: matriz ``len(histórico) × n_jogos`` com a contribuição de cada jogo
            do histórico para o score de cada candidato.
        candidates: índices dos jogos recomendados, já ordenados.
    """
    scores = contributions[:, candidates].sum(axis=0)
    best = contributions[:, candidates].argmax(axis=0)
    recs = []
    for cand, score, b in zip(candidates, scores, best, strict=True):
        source = history_ids[int(b)]
        reason = template.format(name=names.get(source, f"o jogo {source}"))
        recs.append(Recommendation(int(game_ids[cand]), float(score), reason))
    return recs
