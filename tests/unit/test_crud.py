from app.db import crud


def test_usuarios(seeded_db, db_session):
    user = crud.create_user(db_session, steam_real_id=42)

    assert user.id == 8
    assert crud.get_user(db_session, 8) is user
    assert crud.get_user_by_steam_id(db_session, 42) is user
    assert crud.get_user(db_session, 999) is None
    assert [u.id for u in crud.list_users(db_session, limit=2, offset=1)] == [2, 3]


def test_jogos_busca_e_nome(seeded_db, db_session):
    assert [g.id for g in crud.list_games(db_session, search="FALLOUT")] == [1, 2, 9]
    assert [g.id for g in crud.list_games(db_session, limit=2, offset=3)] == [4, 5]
    assert crud.get_game_by_name(db_session, "  dota 2 ").id == 4
    assert crud.get_game_by_name(db_session, "Dota") is None

    game = crud.create_game(db_session, "Fallout 76", ["rpg"])
    assert crud.get_game(db_session, game.id).tags == ["rpg"]


def test_get_games_by_ids_e_missing(seeded_db, db_session):
    assert set(crud.get_games_by_ids(db_session, [1, 4, 99])) == {1, 4}
    assert crud.get_games_by_ids(db_session, []) == {}
    assert crud.missing_game_ids(db_session, [1, 99, 100]) == {99, 100}


def test_interacoes_upsert_e_delete(seeded_db, db_session):
    history = crud.get_interactions(db_session, 2)
    assert [(i.game_id, i.game.name) for i in history] == [
        (1, "Fallout 4"),
        (2, "Fallout New Vegas"),
    ]

    created = crud.upsert_interaction(db_session, 2, 9, hours=3.0)
    assert (created.purchased, created.hours, created.disliked) == (True, 3.0, False)

    updated = crud.upsert_interaction(db_session, 2, 1, disliked=True, hours=None)
    assert (updated.hours, updated.disliked) == (50.0, True)

    assert crud.delete_interactions(db_session, 2, [9]) == 1
    assert [i.game_id for i in crud.get_interactions(db_session, 2)] == [1, 2]
    assert crud.delete_interactions(db_session, 2) == 2
    assert crud.get_interactions(db_session, 2) == []
