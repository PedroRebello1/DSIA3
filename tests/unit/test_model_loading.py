import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app, load_model
from app.services.recommender_service import RecommenderService
from recsys.models.hybrid import HybridRecommender

TRAINED_AT = "2026-11-26T12:00:00+00:00"


@pytest.fixture
def artifact(synthetic_dataset, isolated_artifacts):
    model = HybridRecommender()
    model.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    model.metadata_ = {"trained_at": TRAINED_AT, "params": {"alpha": 0.5}}
    path = get_settings().model_path
    model.save(str(path))
    return path


def test_model_path_vem_da_configuracao(isolated_artifacts):
    assert get_settings().model_path == isolated_artifacts / "model.joblib"


def test_from_artifact_le_versao_e_alpha(artifact):
    service = RecommenderService.from_artifact(artifact)

    assert service.strategy == "hybrid"
    assert service.version == TRAINED_AT
    assert service.alpha == 0.5
    assert service.similar_model is service.model


def test_load_model_sem_artefato_ou_corrompido(isolated_artifacts):
    path = isolated_artifacts / "model.joblib"
    assert load_model(path) is None

    path.parent.mkdir(parents=True)
    path.write_bytes(b"isto nao e um modelo")
    assert load_model(path) is None


def test_api_carrega_o_modelo_no_startup(artifact):
    with TestClient(create_app()) as client:
        health = client.get("/health").json()

    assert health["model_loaded"] is True
    assert health["model_version"] == TRAINED_AT


def test_recomendacoes_usam_o_hibrido(artifact, app, seeded_db):
    with TestClient(app) as client:
        body = client.get("/users/1/recommendations", params={"k": 3}).json()
        info = client.get("/model/info").json()

    assert body["strategy"] == "hybrid"
    assert body["model_version"] == TRAINED_AT
    assert not {1, 2, 3} & {r["game_id"] for r in body["recommendations"]}
    assert info["loaded"] is True
    assert info["strategy"] == "hybrid"
