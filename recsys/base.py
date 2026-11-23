"""Contrato comum a todos os recomendadores do projeto.

Os modelos recebem o histórico do usuário (``{game_id: score}``) em vez do ``user_id``.
Assim o modelo fica desacoplado do banco e usuários novos ou recém-atualizados
recebem recomendações sem retreino.

Formato esperado em ``fit``:

- ``interactions``: ``DataFrame`` com as colunas ``user_id``, ``game_id`` e ``score``
  (score de confiança, ver ``recsys.data``);
- ``games``: ``DataFrame`` com as colunas ``game_id``, ``name`` e, opcionalmente, ``tags``
  (lista de strings).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import joblib
import pandas as pd

USER_COL = "user_id"
GAME_COL = "game_id"
SCORE_COL = "score"
NAME_COL = "name"
TAGS_COL = "tags"


@dataclass
class Recommendation:
    """Um jogo recomendado, com sua pontuação e o motivo (explicabilidade)."""

    game_id: int
    score: float
    reason: str | None = None


@runtime_checkable
class Recommender(Protocol):
    """Interface que a API usa para conversar com qualquer modelo."""

    def fit(self, interactions: pd.DataFrame, games: pd.DataFrame) -> None: ...

    def recommend(
        self,
        history: dict[int, float],
        k: int = 10,
        exclude: set[int] | None = None,
        disliked: set[int] | None = None,
    ) -> list[Recommendation]: ...

    def similar_games(self, game_id: int, k: int = 10) -> list[Recommendation]: ...

    def add_game(self, game_id: int, name: str, tags: list[str] | None = None) -> None: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> Recommender: ...


class BaseRecommender:
    """Implementação base com persistência via joblib e utilitários comuns."""

    def save(self, path: str) -> None:
        """Serializa o modelo em ``path``, criando as pastas necessárias."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> Self:
        """Carrega um modelo salvo com ``save`` e confere o tipo."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} contém {type(model).__name__}, esperado {cls.__name__}")
        return model

    @staticmethod
    def _blocked(exclude: set[int] | None, disliked: set[int] | None) -> set[int]:
        """Jogos que não podem aparecer na recomendação.

        Jogos do histórico só são excluídos se vierem em ``exclude``: quem decide
        (por exemplo, o parâmetro ``exclude_owned`` da API) é quem chama o modelo.
        """
        return (exclude or set()) | (disliked or set())
