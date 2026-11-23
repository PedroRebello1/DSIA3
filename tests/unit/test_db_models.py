import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Base, Game, Interaction, ModelRun, User
from app.db.session import get_db, make_engine


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def test_cria_usuario_jogo_e_interacao(session):
    user = User(steam_real_id=151603712)
    game = Game(name="Fallout 4", tags=["rpg", "mundo aberto"])
    session.add_all([user, game])
    session.flush()
    session.add(Interaction(user_id=user.id, game_id=game.id, hours=87.0))
    session.commit()

    inter = session.scalars(select(Interaction)).one()
    assert inter.purchased is True
    assert inter.disliked is False
    assert inter.hours == 87.0
    assert inter.game.tags == ["rpg", "mundo aberto"]
    assert user.created_at is not None


def test_nome_do_jogo_e_unico(session):
    session.add_all([Game(name="Dota 2"), Game(name="Dota 2")])
    with pytest.raises(IntegrityError):
        session.commit()


def test_remover_usuario_remove_interacoes(session):
    user, game = User(), Game(name="Portal 2")
    session.add_all([user, game])
    session.flush()
    session.add(Interaction(user_id=user.id, game_id=game.id))
    session.commit()

    session.delete(user)
    session.commit()

    assert session.scalars(select(Interaction)).all() == []


def test_model_run_guarda_json(session):
    session.add(ModelRun(params={"alpha": 1.0}, metrics={"ndcg@10": 0.1}, artifact_path="a.joblib"))
    session.commit()

    run = session.scalars(select(ModelRun)).one()
    assert run.params == {"alpha": 1.0}
    assert run.trained_at is not None


def test_get_db_fecha_a_sessao():
    gen = get_db()
    db = next(gen)
    assert db.is_active
    gen.close()


def test_migracao_sobe_e_desce(tmp_path):
    url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")
    tables = set(inspect(make_engine(url)).get_table_names())
    assert {"users", "games", "interactions", "model_runs"} <= tables

    command.downgrade(cfg, "base")
    assert set(inspect(make_engine(url)).get_table_names()) == {"alembic_version"}
