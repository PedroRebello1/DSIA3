"""Fixtures compartilhadas: banco de teste, cliente HTTP e dataset sintético.

Dataset sintético (pequeno e com estrutura conhecida):

- Jogos 1–3 (RPG: Fallout 4, Fallout New Vegas, Skyrim) são jogados juntos
  pelos usuários 1–3;
- Jogos 4–6 (tiro/MOBA: Dota 2, Team Fortress 2, Counter-Strike) pelos usuários 4–6;
- Jogos 7–8 (Portal 2, Half-Life 2) pelo usuário 7, que também tem Team Fortress 2;
- Jogo 9 (Fallout 3) existe no catálogo, mas ninguém jogou.
"""

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.session import get_db
from app.main import create_app
from recsys.data import Dataset, confidence_score
from scripts.seed_db import seed

GAMES = [
    (1, "Fallout 4", ["rpg", "mundo aberto"]),
    (2, "Fallout New Vegas", ["rpg", "mundo aberto"]),
    (3, "The Elder Scrolls V Skyrim", ["rpg", "fantasia"]),
    (4, "Dota 2", ["moba"]),
    (5, "Team Fortress 2", ["tiro"]),
    (6, "Counter-Strike", ["tiro"]),
    (7, "Portal 2", ["puzzle"]),
    (8, "Half-Life 2", ["tiro", "história"]),
    (9, "Fallout 3", ["rpg"]),
]

# (user_id, game_id, horas)
INTERACTIONS = [
    (1, 1, 120.0), (1, 2, 80.0), (1, 3, 200.0),
    (2, 1, 50.0), (2, 2, 30.0),
    (3, 2, 10.0), (3, 3, 0.0),
    (4, 4, 900.0), (4, 5, 40.0), (4, 6, 12.0),
    (5, 4, 300.0), (5, 6, 25.0),
    (6, 5, 2.0), (6, 6, 0.0),
    (7, 5, 15.0), (7, 7, 20.0), (7, 8, 30.0),
]  # fmt: skip


def make_synthetic_dataset(alpha: float = 1.0) -> Dataset:
    """Monta o ``Dataset`` sintético descrito no topo deste arquivo."""
    interactions = pd.DataFrame(INTERACTIONS, columns=["user_id", "game_id", "hours"])
    interactions["score"] = confidence_score(interactions["hours"].to_numpy(), alpha)
    games = pd.DataFrame(GAMES, columns=["game_id", "name", "tags"])
    users = pd.DataFrame(
        {
            "user_id": sorted(interactions["user_id"].unique()),
            "steam_real_id": [1000 + u for u in sorted(interactions["user_id"].unique())],
        }
    )
    return Dataset(interactions=interactions, games=games, users=users)


@pytest.fixture
def synthetic_dataset() -> Dataset:
    """Dataset sintético pequeno, com grupos de jogos bem separados."""
    return make_synthetic_dataset()


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Banco SQLite em memória, compartilhado entre threads (necessário para o TestClient)."""
    # StaticPool: uma única conexão, então os dados persistem entre sessões.
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def db_session(session_factory) -> Iterator[Session]:
    """Sessão avulsa para preparar ou conferir dados direto no banco."""
    with session_factory() as session:
        yield session


@pytest.fixture
def app(session_factory) -> FastAPI:
    """Aplicação com ``get_db`` apontando para o banco de teste."""
    application = create_app()

    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    application.dependency_overrides[get_db] = override_get_db
    return application


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded_db(engine, synthetic_dataset) -> Dataset:
    """Popula o banco de teste com o dataset sintético (via ``scripts/seed_db.py``)."""
    seed(engine, synthetic_dataset)
    return synthetic_dataset
