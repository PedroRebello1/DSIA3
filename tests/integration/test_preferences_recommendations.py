"""Integração de preferências e recomendações, incluindo o fluxo completo do usuário.

Popularidade no dataset sintético (usuários distintos por jogo):
2, 5 e 6 → 3 · 1, 3 e 4 → 2 · 7 e 8 → 1 · 9 → 0.
"""

import pytest

pytestmark = pytest.mark.usefixtures("seeded_db")


def history(client, user_id):
    user = client.get(f"/users/{user_id}").json()
    return {h["game_id"]: (h["hours"], h["disliked"]) for h in user["history"]}


def rec_ids(client, user_id, k=3):
    response = client.get(f"/users/{user_id}/recommendations", params={"k": k})
    assert response.status_code == 200
    return [r["game_id"] for r in response.json()["recommendations"]]


# ---- Preferências ----


def test_put_substitui_o_historico(client):
    response = client.put(
        "/users/1/preferences", json={"games": [{"game_id": 7, "hours": 3}, {"game_id": 8}]}
    )

    assert response.status_code == 200
    assert [h["game_id"] for h in response.json()["history"]] == [7, 8]
    assert history(client, 1) == {7: (3.0, False), 8: (0.0, False)}


def test_patch_atualiza_adiciona_e_remove(client):
    response = client.patch(
        "/users/2/preferences",
        json={
            "upsert": [{"game_id": 1, "disliked": True}, {"game_id": 9, "hours": 4}],
            "remove": [2],
        },
    )

    assert response.status_code == 200
    assert history(client, 2) == {1: (50.0, True), 9: (4.0, False)}


@pytest.mark.parametrize(
    ("method", "path", "payload", "status", "code"),
    [
        ("put", "/users/99/preferences", {"games": []}, 404, "not_found"),
        ("patch", "/users/99/preferences", {"remove": [1]}, 404, "not_found"),
        ("put", "/users/1/preferences", {"games": [{"game_id": 999}]}, 404, "not_found"),
        ("patch", "/users/1/preferences", {"upsert": [{"game_id": 999}]}, 404, "not_found"),
        ("put", "/users/1/preferences", {}, 422, "validation_error"),
        ("patch", "/users/1/preferences", {}, 422, "validation_error"),
        ("patch", "/users/1/preferences", {"upsert": [{"game_id": 1}], "remove": [1]}, 422, None),
        ("patch", "/users/1/preferences", {"upsert": [{"game_id": 1, "hours": -2}]}, 422, None),
    ],
)
def test_erros_de_preferencias(client, method, path, payload, status, code):
    response = getattr(client, method)(path, json=payload)

    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == (code or "validation_error")
    assert error["message"]


def test_erro_de_validacao_aponta_o_campo(client):
    response = client.patch("/users/1/preferences", json={"upsert": [{"game_id": 1, "hours": -2}]})

    fields = [d["field"] for d in response.json()["error"]["details"]]
    assert fields == ["body.upsert.0.hours"]


def test_erro_nao_altera_o_historico(client):
    before = history(client, 1)
    client.put("/users/1/preferences", json={"games": [{"game_id": 7}, {"game_id": 999}]})

    assert history(client, 1) == before


# ---- Recomendações ----


def test_recomendacoes_de_um_usuario(client):
    response = client.get("/users/1/recommendations", params={"k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["strategy"] == "popular"
    assert body["model_version"]
    assert body["recommendations"] == [
        {
            "game_id": 5,
            "name": "Team Fortress 2",
            "score": 1.0,
            "reason": "Popular entre jogadores",
        },
        {"game_id": 6, "name": "Counter-Strike", "score": 1.0, "reason": "Popular entre jogadores"},
    ]


def test_k_padrao_e_limite_do_catalogo(client):
    recs = client.get("/users/1/recommendations").json()["recommendations"]

    # k padrão é 10, mas só sobram 6 jogos fora do histórico do usuário 1.
    assert [r["game_id"] for r in recs] == [5, 6, 4, 7, 8, 9]


@pytest.mark.parametrize(("path", "status"), [("/users/99", 404), ("/users/1", 422)])
def test_erros_de_recomendacao(client, path, status):
    params = {"k": 0} if status == 422 else {}
    response = client.get(f"{path}/recommendations", params=params)

    assert response.status_code == status
    assert "error" in response.json()


def test_health_indica_modelo_carregado_apos_primeira_recomendacao(client):
    assert client.get("/health").json()["model_loaded"] is False
    client.get("/users/1/recommendations")

    health = client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["model_version"]


# ---- Fluxo completo ----


def test_fluxo_completo_recomendacoes_mudam_com_as_preferencias(client):
    # 1. Usuário novo (cold start): recebe os mais populares.
    user_id = client.post("/users", json={}).json()["id"]
    assert rec_ids(client, user_id) == [2, 5, 6]

    # 2. Adiciona jogos: eles saem das recomendações.
    client.patch(
        f"/users/{user_id}/preferences",
        json={"upsert": [{"game_id": 2, "hours": 30}, {"game_id": 5}]},
    )
    assert rec_ids(client, user_id) == [6, 1, 3]

    # 3. Marca um jogo como "não gostei": ele também não é recomendado.
    client.patch(
        f"/users/{user_id}/preferences", json={"upsert": [{"game_id": 6, "disliked": True}]}
    )
    assert rec_ids(client, user_id) == [1, 3, 4]

    # 4. Substitui o histórico: os jogos removidos voltam a ser recomendados.
    client.put(f"/users/{user_id}/preferences", json={"games": [{"game_id": 7}]})
    assert rec_ids(client, user_id) == [2, 5, 6]
    assert history(client, user_id) == {7: (0.0, False)}
