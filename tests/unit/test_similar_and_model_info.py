from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import NotFoundError
from app.db import crud
from app.db.models import ModelRun
from app.services.model_info import get_model_info
from app.services.recommender_service import RecommenderService
from recsys.models.content import ContentRecommender


@pytest.fixture
def service(seeded_db, db_session):
    return RecommenderService.from_db(db_session)


def test_similares_usam_o_modelo_de_conteudo(service, db_session):
    assert isinstance(service.similar_model, ContentRecommender)

    out = service.similar_games(db_session, 1, k=2)

    assert (out.game_id, out.name) == (1, "Fallout 4")
    assert {g.name for g in out.similar} == {"Fallout New Vegas", "Fallout 3"}


def test_similares_de_jogo_inexistente(service, db_session):
    with pytest.raises(NotFoundError):
        service.similar_games(db_session, 999, k=3)


def test_similar_model_padrao_e_o_proprio_modelo():
    model = object()
    assert RecommenderService(model, "x", "v").similar_model is model


def test_latest_model_run(db_session):
    assert crud.get_latest_model_run(db_session) is None
    now = datetime.now(UTC)
    db_session.add_all(
        [
            ModelRun(trained_at=now - timedelta(days=1), params={}, metrics={}, artifact_path="a"),
            ModelRun(trained_at=now, params={"k": 50}, metrics={"ndcg@10": 0.3}, artifact_path="b"),
        ]
    )
    db_session.commit()

    assert crud.get_latest_model_run(db_session).artifact_path == "b"


def test_model_info_sem_modelo_e_sem_treino(db_session):
    info = get_model_info(db_session, None)

    assert info.loaded is False
    assert info.strategy is None
    assert info.last_run is None


def test_model_info_com_modelo_e_treino(service, db_session):
    db_session.add(ModelRun(params={"k": 50}, metrics={"ndcg@10": 0.35}, artifact_path="m.joblib"))
    db_session.commit()

    info = get_model_info(db_session, service)

    assert info.loaded is True
    assert info.strategy == "popular"
    assert info.version == service.version
    assert info.last_run.metrics == {"ndcg@10": 0.35}
