import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Base, Game, Interaction, User
from app.db.session import make_engine
from recsys.data import Dataset
from scripts.seed_db import seed


@pytest.fixture
def engine():
    eng = make_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def dataset():
    return Dataset(
        interactions=pd.DataFrame(
            {"user_id": [1, 1, 2], "game_id": [10, 20, 10], "hours": [5.0, 0.0, 1.5]}
        ).assign(score=1.0),
        games=pd.DataFrame(
            {"game_id": [10, 20], "name": ["Fallout 4", "Dota 2"], "tags": [[], []]}
        ),
        users=pd.DataFrame({"user_id": [1, 2], "steam_real_id": [151603712, 187131847]}),
    )


def test_seed_insere_tudo(engine, dataset):
    counts = seed(engine, dataset)

    assert counts == {"users": 2, "games": 2, "interactions": 3}
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Interaction)) == 3
        assert db.get(User, 2).steam_real_id == 187131847
        assert db.get(Game, 20).name == "Dota 2"
        inter = db.get(Interaction, (1, 10))
        assert inter.hours == 5.0
        assert inter.purchased is True
        assert inter.disliked is False


def test_seed_sem_tabelas():
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        seed(make_engine("sqlite://"), None)
