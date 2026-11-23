"""Carga e limpeza do dataset Steam (caserec) e construção da matriz de interações.

Arquivos esperados em ``data/raw/steam/`` (TSV com cabeçalho):

- ``game_purchase.dat``: ``User_ID``, ``Game_ID``, ``Purchase`` (sempre 1.0);
- ``game_play.dat``: ``User_ID``, ``Game_ID``, ``Hours``;
- ``item_info.dat``: ``Game_ID``, ``Game Name``;
- ``user_info.dat``: ``New_ID``, ``Real_ID``.

Score de confiança de cada par (usuário, jogo)::

    score = 1 + alpha * log1p(horas)    # horas = 0 quando só houve compra
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.base import GAME_COL, NAME_COL, SCORE_COL, TAGS_COL, USER_COL

DEFAULT_DATA_DIR = Path("data/raw/steam")
DEFAULT_ALPHA = 1.0
HOURS_COL = "hours"

PURCHASE_FILE = "game_purchase.dat"
PLAY_FILE = "game_play.dat"
GAMES_FILE = "item_info.dat"
USERS_FILE = "user_info.dat"


@dataclass
class Dataset:
    """Dados limpos, prontos para o modelo e para o seed do banco.

    Attributes:
        interactions: ``user_id``, ``game_id``, ``hours``, ``score`` (um par por linha).
        games: ``game_id``, ``name``, ``tags`` (lista vazia no dataset original).
        users: ``user_id``, ``steam_real_id``.
    """

    interactions: pd.DataFrame
    games: pd.DataFrame
    users: pd.DataFrame


@dataclass
class InteractionMatrix:
    """Matriz esparsa ``usuários × jogos`` com o mapeamento entre IDs e índices."""

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    game_ids: np.ndarray

    def __post_init__(self) -> None:
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.game_index = {int(g): j for j, g in enumerate(self.game_ids)}

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def sparsity(self) -> float:
        """Fração de células vazias da matriz."""
        rows, cols = self.matrix.shape
        total = rows * cols
        return 1.0 - self.matrix.nnz / total if total else 0.0


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _apply_aliases(df: pd.DataFrame, aliases: dict[int, int] | None) -> pd.DataFrame:
    if aliases:
        df = df.copy()
        df[GAME_COL] = df[GAME_COL].replace(aliases)
    return df


def clean_purchases(purchases: pd.DataFrame, aliases: dict[int, int] | None = None) -> pd.DataFrame:
    """Padroniza colunas, unifica IDs de jogos repetidos e remove pares repetidos."""
    df = purchases.rename(columns={"User_ID": USER_COL, "Game_ID": GAME_COL})
    df = _apply_aliases(df[[USER_COL, GAME_COL]], aliases)
    return df.drop_duplicates().reset_index(drop=True)


def clean_plays(plays: pd.DataFrame, aliases: dict[int, int] | None = None) -> pd.DataFrame:
    """Padroniza colunas, unifica IDs de jogos repetidos e soma as horas de pares repetidos.

    No dataset os pares repetidos são registros separados de sessão do mesmo jogo,
    por isso as horas são somadas. Horas negativas ou ausentes viram 0.
    """
    df = plays.rename(columns={"User_ID": USER_COL, "Game_ID": GAME_COL, "Hours": HOURS_COL})
    df = _apply_aliases(df, aliases)
    df[HOURS_COL] = pd.to_numeric(df[HOURS_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df.groupby([USER_COL, GAME_COL], as_index=False)[HOURS_COL].sum()


def _standardize_games(games: pd.DataFrame) -> pd.DataFrame:
    df = games.rename(columns={"Game_ID": GAME_COL, "Game Name": NAME_COL})
    df = df[[GAME_COL, NAME_COL]].drop_duplicates(GAME_COL).copy()
    df[NAME_COL] = df[NAME_COL].astype(str).str.strip()
    return df.sort_values(GAME_COL).reset_index(drop=True)


def game_id_aliases(games: pd.DataFrame) -> dict[int, int]:
    """Mapeia IDs de jogos com nome repetido (após tirar espaços) para o menor ID.

    O catálogo original tem títulos que só diferem por um espaço no fim
    (ex.: ``"Universe Sandbox"`` e ``"Universe Sandbox "``); eles são o mesmo jogo.
    """
    df = _standardize_games(games)
    canonical = df.groupby(NAME_COL)[GAME_COL].transform("min")
    repeated = canonical != df[GAME_COL]
    return {
        int(g): int(c) for g, c in zip(df.loc[repeated, GAME_COL], canonical[repeated], strict=True)
    }


def clean_games(games: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o catálogo: tira espaços dos nomes, mantém um ID por nome e cria ``tags``."""
    df = _standardize_games(games).drop_duplicates(NAME_COL).reset_index(drop=True)
    df[TAGS_COL] = [[] for _ in range(len(df))]
    return df


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o mapeamento de IDs de usuário."""
    df = users.rename(columns={"New_ID": USER_COL, "Real_ID": "steam_real_id"})
    df = df[[USER_COL, "steam_real_id"]].drop_duplicates(USER_COL)
    return df.sort_values(USER_COL).reset_index(drop=True)


def confidence_score(hours: pd.Series | np.ndarray | float, alpha: float = DEFAULT_ALPHA):
    """Score de confiança ``1 + alpha * log1p(horas)``; vale 1 para compra sem horas."""
    if alpha < 0:
        raise ValueError("alpha deve ser >= 0")
    return 1.0 + alpha * np.log1p(np.clip(hours, 0.0, None))


def build_interactions(
    purchases: pd.DataFrame, plays: pd.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """Une compras e horas (já limpas) em um par por linha, com o score de confiança.

    Horas registradas sem compra correspondente também viram interação: ter jogado
    implica ter o jogo.
    """
    df = purchases.merge(plays, on=[USER_COL, GAME_COL], how="outer")
    df[HOURS_COL] = df[HOURS_COL].fillna(0.0)
    df[SCORE_COL] = confidence_score(df[HOURS_COL].to_numpy(), alpha)
    df = df.sort_values([USER_COL, GAME_COL]).reset_index(drop=True)
    return df[[USER_COL, GAME_COL, HOURS_COL, SCORE_COL]]


def load_dataset(data_dir: str | Path = DEFAULT_DATA_DIR, alpha: float = DEFAULT_ALPHA) -> Dataset:
    """Lê os quatro arquivos ``.dat`` e devolve o ``Dataset`` limpo."""
    data_dir = Path(data_dir)
    missing = [
        f for f in (PURCHASE_FILE, PLAY_FILE, GAMES_FILE, USERS_FILE) if not (data_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em {data_dir}: {', '.join(missing)}. "
            "Rode scripts/download_data.sh primeiro."
        )

    raw_games = _read_tsv(data_dir / GAMES_FILE)
    aliases = game_id_aliases(raw_games)
    purchases = clean_purchases(_read_tsv(data_dir / PURCHASE_FILE), aliases)
    plays = clean_plays(_read_tsv(data_dir / PLAY_FILE), aliases)
    return Dataset(
        interactions=build_interactions(purchases, plays, alpha),
        games=clean_games(raw_games),
        users=clean_users(_read_tsv(data_dir / USERS_FILE)),
    )


def build_matrix(
    interactions: pd.DataFrame,
    user_ids: np.ndarray | None = None,
    game_ids: np.ndarray | None = None,
    value_col: str = SCORE_COL,
) -> InteractionMatrix:
    """Monta a matriz esparsa ``usuários × jogos`` (CSR, float32).

    Se ``user_ids``/``game_ids`` forem dados, a matriz usa essa ordem (útil para incluir
    jogos sem interação); pares com IDs fora dessas listas são ignorados.
    """
    if user_ids is None:
        user_ids = np.sort(interactions[USER_COL].unique())
    if game_ids is None:
        game_ids = np.sort(interactions[GAME_COL].unique())
    user_ids = np.asarray(user_ids, dtype=np.int64)
    game_ids = np.asarray(game_ids, dtype=np.int64)

    rows = pd.Index(user_ids).get_indexer(interactions[USER_COL])
    cols = pd.Index(game_ids).get_indexer(interactions[GAME_COL])
    keep = (rows >= 0) & (cols >= 0)
    values = interactions[value_col].to_numpy(dtype=np.float32)[keep]

    matrix = sparse.csr_matrix(
        (values, (rows[keep], cols[keep])), shape=(len(user_ids), len(game_ids)), dtype=np.float32
    )
    matrix.sum_duplicates()
    return InteractionMatrix(matrix=matrix, user_ids=user_ids, game_ids=game_ids)
