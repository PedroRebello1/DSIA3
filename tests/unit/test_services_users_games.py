import pytest

from app.core.errors import ConflictError, NotFoundError
from app.schemas.game import GameCreate
from app.schemas.user import UserCreate
from app.services import games, users


def test_create_user_com_jogos_iniciais(seeded_db, db_session):
    detail = users.create_user(
        db_session, UserCreate(games=[{"game_id": 1, "hours": 10}, {"game_id": 4}])
    )

    assert detail.id == 8
    assert [(h.game_id, h.name, h.hours) for h in detail.history] == [
        (1, "Fallout 4", 10.0),
        (4, "Dota 2", 0.0),
    ]


def test_create_user_rejeita_jogo_inexistente_e_steam_id_repetido(seeded_db, db_session):
    with pytest.raises(NotFoundError, match=r"\[99\]"):
        users.create_user(db_session, UserCreate(games=[{"game_id": 99}]))
    with pytest.raises(ConflictError):
        users.create_user(db_session, UserCreate(steam_real_id=1001))


def test_get_user_e_list_users(seeded_db, db_session):
    assert len(users.get_user(db_session, 1).history) == 3
    assert [u.id for u in users.list_users(db_session, limit=3, offset=0)] == [1, 2, 3]
    with pytest.raises(NotFoundError, match="Usuário 99"):
        users.get_user(db_session, 99)


def test_create_game_e_conflito_de_nome(seeded_db, db_session):
    game = games.create_game(db_session, GameCreate(name="Hades", tags=["roguelike"]))

    assert games.get_game(db_session, game.id).name == "Hades"
    with pytest.raises(ConflictError, match="Hades"):
        games.create_game(db_session, GameCreate(name="  HADES "))


def test_get_game_inexistente_e_busca(seeded_db, db_session):
    with pytest.raises(NotFoundError, match="Jogo 99"):
        games.get_game(db_session, 99)
    assert [g.name for g in games.list_games(db_session, "half", 10, 0)] == ["Half-Life 2"]
