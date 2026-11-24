import pytest
from pydantic import ValidationError

from app.schemas.game import GameCreate
from app.schemas.interaction import InteractionIn
from app.schemas.user import UserCreate


def test_game_create_normaliza_nome_e_tags():
    game = GameCreate(name="  Hades  ", tags=[" Roguelike ", "AÇÃO"])

    assert game.name == "Hades"
    assert game.tags == ["roguelike", "ação"]


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
