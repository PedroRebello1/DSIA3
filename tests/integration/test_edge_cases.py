"""Casos de borda que dependem de condições difíceis de reproduzir pela API."""

import pytest

from app.db import crud

pytestmark = pytest.mark.usefixtures("seeded_db")


def test_corrida_no_cadastro_de_jogo_retorna_409(client, monkeypatch):
    # Simula outra requisição criando o mesmo nome entre a checagem e o INSERT.
    monkeypatch.setattr(crud, "get_game_by_name", lambda db, name: None)

    response = client.post("/games", json={"name": "Dota 2"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"
    # A sessão foi desfeita: a API continua funcionando normalmente.
    assert client.get("/games/4").json()["name"] == "Dota 2"
