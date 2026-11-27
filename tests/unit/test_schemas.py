import pytest
from pydantic import ValidationError

from app.schemas.game import GameCreate
from app.schemas.interaction import InteractionIn
from app.schemas.preference import PreferencesPatch
from app.schemas.user import UserCreate


def test_game_create_normaliza_nome_e_tags():
    game = GameCreate(name="  Hades  ", tags=[" Roguelike ", "AÇÃO"])

    assert game.name == "Hades"
    assert game.tags == ["roguelike", "ação"]


def test_game_create_remove_tags_repetidas():
    game = GameCreate(name="Hades", tags=["Roguelike", "ação", " roguelike", "AÇÃO", "indie"])

    assert game.tags == ["roguelike", "ação", "indie"]


@pytest.mark.parametrize("name", ["", "   ", "x" * 256])
def test_game_create_rejeita_nome_invalido(name):
    with pytest.raises(ValidationError):
        GameCreate(name=name)


def test_interaction_padroes_e_limites():
    item = InteractionIn(game_id=1)
    assert (item.purchased, item.hours, item.disliked) == (True, 0.0, False)

    with pytest.raises(ValidationError):
        InteractionIn(game_id=1, hours=-1)
    with pytest.raises(ValidationError):
        InteractionIn(game_id=0)


def test_user_create_rejeita_jogo_repetido():
    with pytest.raises(ValidationError, match="jogos repetidos"):
        UserCreate(games=[{"game_id": 1}, {"game_id": 2}, {"game_id": 1}])

    assert UserCreate().games == []


@pytest.mark.parametrize("remove", [[0], [-3], [1, 0]])
def test_patch_rejeita_ids_invalidos_em_remove(remove):
    with pytest.raises(ValidationError):
        PreferencesPatch(remove=remove)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/users/0", None),
        ("get", "/users/-1/recommendations", None),
        ("patch", "/users/0/preferences", {"remove": [1]}),
        ("get", "/games/0", None),
        ("get", "/games/-5/similar", None),
    ],
)
def test_ids_no_caminho_precisam_ser_positivos(client, seeded_db, method, path, body):
    response = client.request(method, path, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"].startswith("path.")
