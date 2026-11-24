"""Integração dos endpoints de usuários e jogos (HTTP + banco de teste)."""

import pytest

pytestmark = pytest.mark.usefixtures("seeded_db")


# ---- Jogos ----


def test_cria_jogo_e_consulta(client):
    response = client.post("/games", json={"name": "  Hades ", "tags": ["Roguelike"]})

    assert response.status_code == 201
    game = response.json()
    assert game["name"] == "Hades"
    assert game["tags"] == ["roguelike"]
    assert client.get(f"/games/{game['id']}").json()["name"] == "Hades"


def test_cria_jogo_com_nome_repetido_retorna_409(client):
    response = client.post("/games", json={"name": "dota 2"})

    assert response.status_code == 409
    assert "Dota 2" in response.text


@pytest.mark.parametrize("payload", [{}, {"name": ""}, {"name": "x", "tags": "rpg"}])
def test_cria_jogo_invalido_retorna_422(client, payload):
    assert client.post("/games", json=payload).status_code == 422


def test_busca_jogos_por_nome(client):
    response = client.get("/games", params={"search": "fallout"})

    assert response.status_code == 200
    assert [g["name"] for g in response.json()] == ["Fallout 4", "Fallout New Vegas", "Fallout 3"]


def test_lista_jogos_com_paginacao(client):
    page = client.get("/games", params={"limit": 3, "offset": 3}).json()

    assert [g["id"] for g in page] == [4, 5, 6]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_paginacao_invalida_retorna_422(client, params):
    assert client.get("/games", params=params).status_code == 422


def test_jogo_inexistente_retorna_404(client):
    response = client.get("/games/999")

    assert response.status_code == 404
    assert "não encontrado" in response.text


# ---- Usuários ----


def test_cria_usuario_sem_historico(client):
    response = client.post("/users", json={})

    assert response.status_code == 201
    user = response.json()
    assert user["id"] == 8
    assert user["steam_real_id"] is None
    assert user["history"] == []


def test_cria_usuario_com_jogos_e_consulta_o_historico(client):
    payload = {"steam_real_id": 76561198, "games": [{"game_id": 7, "hours": 12.5}, {"game_id": 9}]}
    user_id = client.post("/users", json=payload).json()["id"]

    user = client.get(f"/users/{user_id}").json()

    assert user["steam_real_id"] == 76561198
    assert [(h["game_id"], h["name"], h["hours"]) for h in user["history"]] == [
        (7, "Portal 2", 12.5),
        (9, "Fallout 3", 0.0),
    ]


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"games": [{"game_id": 999}]}, 404),
        ({"steam_real_id": 1001}, 409),
        ({"games": [{"game_id": 1, "hours": -1}]}, 422),
        ({"games": [{"game_id": 1}, {"game_id": 1}]}, 422),
        ({"steam_real_id": "abc"}, 422),
    ],
)
def test_cria_usuario_com_erro(client, payload, status):
    assert client.post("/users", json=payload).status_code == status


def test_erro_na_criacao_nao_deixa_usuario_pela_metade(client):
    client.post("/users", json={"games": [{"game_id": 1}, {"game_id": 999}]})

    assert len(client.get("/users", params={"limit": 100}).json()) == 7


def test_lista_usuarios(client):
    users = client.get("/users", params={"limit": 2, "offset": 5}).json()

    assert [u["id"] for u in users] == [6, 7]
    assert "history" not in users[0]


def test_usuario_inexistente_retorna_404(client):
    assert client.get("/users/999").status_code == 404
    assert client.get("/users/abc").status_code == 422
