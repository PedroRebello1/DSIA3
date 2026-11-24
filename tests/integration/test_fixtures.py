from sqlalchemy import func, select

from app.db.models import Game, Interaction, User


def test_seeded_db_popula_o_banco_de_teste(seeded_db, db_session):
    assert db_session.scalar(select(func.count()).select_from(User)) == 7
    assert db_session.scalar(select(func.count()).select_from(Game)) == 9
    assert db_session.scalar(select(func.count()).select_from(Interaction)) == len(
        seeded_db.interactions
    )


def test_banco_e_compartilhado_entre_sessoes(session_factory):
    with session_factory() as s1:
        s1.add(Game(name="Jogo Novo"))
        s1.commit()
    with session_factory() as s2:
        assert s2.scalar(select(Game.name)) == "Jogo Novo"


def test_client_responde(client):
    assert client.get("/health").status_code == 200


def test_dataset_sintetico_tem_jogo_sem_interacao(synthetic_dataset):
    played = set(synthetic_dataset.interactions["game_id"])
    assert 9 in set(synthetic_dataset.games["game_id"]) - played
