import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import ConflictError
from app.services import model_training
from app.services.model_training import is_retraining, retrain_model, start_retrain
from app.services.recommender_service import RecommenderService
from recsys.models.hybrid import HybridRecommender


def test_start_retrain_impede_dois_ao_mesmo_tempo():
    app = FastAPI()
    start_retrain(app)

    assert is_retraining(app)
    with pytest.raises(ConflictError):
        start_retrain(app)


def test_retrain_troca_o_modelo_e_registra(seeded_db, engine, isolated_artifacts):
    app = FastAPI()
    old = object()
    app.state.recommender = old
    start_retrain(app)

    retrain_model(app, engine, isolated_artifacts, search=False)

    service = app.state.recommender
    assert service is not old
    assert isinstance(service.model, HybridRecommender)
    assert service.strategy == "hybrid"
    assert (isolated_artifacts / "model.joblib").exists()
    assert not is_retraining(app)


def test_falha_no_retreino_mantem_o_modelo_anterior(engine, isolated_artifacts, monkeypatch):
    def boom(_engine):
        raise RuntimeError("banco fora do ar")

    monkeypatch.setattr(model_training, "load_from_db", boom)
    app = FastAPI()
    app.state.recommender = "anterior"
    start_retrain(app)

    retrain_model(app, engine, isolated_artifacts, search=False)

    assert app.state.recommender == "anterior"
    assert not is_retraining(app)


def test_api_retreina_em_segundo_plano(app, seeded_db):
    with TestClient(app) as client:
        assert client.get("/model/info").json()["loaded"] is False

        response = client.post("/model/retrain")
        # O TestClient executa as background tasks logo após a resposta.
        info = client.get("/model/info").json()
        recs = client.get("/users/7/recommendations", params={"strategy": "cf"})

    assert response.status_code == 202
    assert response.json()["status"] == "started"
    assert info["loaded"] is True
    assert info["strategy"] == "hybrid"
    assert info["retraining"] is False
    assert info["last_run"]["artifact_path"].endswith("model.joblib")
    assert info["version"][:19] == info["last_run"]["trained_at"][:19]
    assert recs.status_code == 200


def test_retrain_com_um_em_andamento_retorna_409(app):
    with TestClient(app) as client:
        app.state.retraining = True  # simula um retreino já em andamento
        response = client.post("/model/retrain")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_jogo_cadastrado_entra_no_modelo_em_memoria(app, seeded_db, synthetic_dataset):
    model = HybridRecommender()
    model.fit(synthetic_dataset.interactions, synthetic_dataset.games)

    with TestClient(app) as client:
        app.state.recommender = RecommenderService(model, "hybrid", "v1")
        game_id = client.post("/games", json={"name": "Fallout 76", "tags": ["rpg"]}).json()["id"]
        similar = client.get("/games/1/similar", params={"k": 5}).json()["similar"]

    assert game_id in {g["game_id"] for g in similar}
    assert game_id in model.content.game_index_


def test_cadastro_sem_modelo_carregado_nao_falha(client, seeded_db):
    assert client.post("/games", json={"name": "Jogo Novo"}).status_code == 201
