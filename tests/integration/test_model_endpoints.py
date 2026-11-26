"""Integração dos endpoints do modelo: /similar, /model/*, strategy e exclude_owned."""

import pytest
from fastapi.testclient import TestClient

from recsys.train import register_run, train

pytestmark = pytest.mark.usefixtures("seeded_db")


@pytest.fixture
def trained(engine, synthetic_dataset, isolated_artifacts):
    """Treina o híbrido no dataset sintético e grava o artefato onde a API procura."""
    result = train(
        synthetic_dataset.interactions[["user_id", "game_id", "hours"]],
        synthetic_dataset.games,
        artifacts_dir=isolated_artifacts,
        search=False,
        min_interactions=3,
    )
    register_run(engine, result)
    return result


@pytest.fixture
def hybrid_client(trained, app):
    with TestClient(app) as client:
        yield client


def rec_ids(client, user_id, **params):
    response = client.get(f"/users/{user_id}/recommendations", params=params)
    assert response.status_code == 200, response.text
    return [r["game_id"] for r in response.json()["recommendations"]]


# ---- /model/info ----


def test_model_info_sem_modelo_treinado(client):
    info = client.get("/model/info").json()

    assert info == {
        "loaded": False,
        "strategy": None,
        "version": None,
        "last_run": None,
        "retraining": False,
    }


def test_model_info_com_modelo_treinado(hybrid_client, trained):
    info = hybrid_client.get("/model/info").json()

    assert info["loaded"] is True
    assert info["strategy"] == "hybrid"
    assert info["version"] == trained.model.metadata_["trained_at"]
    assert info["last_run"]["params"]["k_neighbors"] == trained.params["k_neighbors"]
    assert "ndcg@10" in info["last_run"]["metrics"]


# ---- /model/retrain ----


def test_retrain_troca_a_versao_e_registra_novo_treino(hybrid_client, trained):
    response = hybrid_client.post("/model/retrain")
    info = hybrid_client.get("/model/info").json()

    assert response.status_code == 202
    assert info["last_run"]["id"] == 2
    assert info["retraining"] is False


def test_retrain_aprende_jogos_e_usuarios_novos(client):
    game_id = client.post("/games", json={"name": "Jogo Cooperativo", "tags": ["coop"]}).json()[
        "id"
    ]
    for _ in range(3):
        client.post("/users", json={"games": [{"game_id": game_id, "hours": 10}, {"game_id": 7}]})

    client.post("/model/retrain")

    # Depois do retreino, o usuário 7 (que tem Portal 2) recebe o jogo novo pelo CF.
    assert game_id in rec_ids(client, 7, strategy="cf", k=9)


# ---- strategy, exclude_owned e reason ----


@pytest.mark.parametrize(
    ("strategy", "reason"),
    [
        ("hybrid", None),
        ("cf", "Porque você jogou"),
        ("content", "Parecido com"),
        ("popular", "Popular entre jogadores"),
    ],
)
def test_estrategias(hybrid_client, strategy, reason):
    body = hybrid_client.get(
        "/users/7/recommendations", params={"strategy": strategy, "k": 3}
    ).json()

    assert body["strategy"] == strategy
    assert body["recommendations"]
    assert all(r["reason"] for r in body["recommendations"])
    if reason:
        assert all(r["reason"].startswith(reason) for r in body["recommendations"])


def test_estrategia_padrao_e_hibrida(hybrid_client):
    assert hybrid_client.get("/users/1/recommendations").json()["strategy"] == "hybrid"


def test_sem_modelo_treinado_so_popular(client):
    assert client.get("/users/1/recommendations", params={"strategy": "cf"}).status_code == 503
    assert rec_ids(client, 1, strategy="popular", k=1) == [5]


def test_estrategia_invalida(hybrid_client):
    response = hybrid_client.get("/users/1/recommendations", params={"strategy": "aleatoria"})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "query.strategy"


def test_exclude_owned(hybrid_client):
    owned = {5, 7, 8}

    assert not owned & set(rec_ids(hybrid_client, 7, k=9))
    assert owned & set(rec_ids(hybrid_client, 7, k=9, exclude_owned="false"))


def test_nao_gostei_nunca_volta_mesmo_sem_exclude_owned(hybrid_client):
    hybrid_client.patch("/users/7/preferences", json={"upsert": [{"game_id": 6, "disliked": True}]})

    assert 6 not in rec_ids(hybrid_client, 7, k=9, exclude_owned="false")


def test_nao_gostei_penaliza_jogos_parecidos(hybrid_client):
    user_id = hybrid_client.post("/users", json={}).json()["id"]
    before = rec_ids(hybrid_client, user_id, k=9)

    hybrid_client.patch(
        f"/users/{user_id}/preferences", json={"upsert": [{"game_id": 4, "disliked": True}]}
    )
    after = rec_ids(hybrid_client, user_id, k=9)

    # Counter-Strike (6) é parecido com Dota 2 (4) e cai no ranking.
    assert after.index(6) > before.index(6)


# ---- /games/{id}/similar ----


def test_similares_com_hibrido(hybrid_client):
    body = hybrid_client.get("/games/1/similar", params={"k": 3}).json()

    assert body["game_id"] == 1
    assert body["name"] == "Fallout 4"
    assert {g["game_id"] for g in body["similar"]} == {2, 3, 9}
    assert all(g["reason"] for g in body["similar"])


def test_similares_sem_modelo_treinado_usam_conteudo(client):
    similar = client.get("/games/1/similar", params={"k": 2}).json()["similar"]

    assert {g["name"] for g in similar} == {"Fallout New Vegas", "Fallout 3"}


@pytest.mark.parametrize(
    ("path", "status"), [("/games/999/similar", 404), ("/games/1/similar?k=0", 422)]
)
def test_similares_erros(hybrid_client, path, status):
    assert hybrid_client.get(path).status_code == status


def test_jogo_novo_aparece_nos_similares_sem_retreino(hybrid_client):
    game_id = hybrid_client.post("/games", json={"name": "Fallout 76", "tags": ["rpg"]}).json()[
        "id"
    ]

    similar = hybrid_client.get("/games/1/similar", params={"k": 5}).json()["similar"]
    recs = rec_ids(hybrid_client, 1, strategy="content", k=5)

    assert game_id in {g["game_id"] for g in similar}
    assert game_id in recs
