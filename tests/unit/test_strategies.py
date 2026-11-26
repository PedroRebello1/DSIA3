import pytest

from app.core.errors import ModelUnavailableError
from app.schemas.recommendation import Strategy
from app.services.recommender_service import RecommenderService
from recsys.models.hybrid import HybridRecommender


@pytest.fixture
def hybrid_service(synthetic_dataset):
    model = HybridRecommender()
    model.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    return RecommenderService(model, "hybrid", "v1")


@pytest.fixture
def popular_service(seeded_db, db_session):
    return RecommenderService.from_db(db_session)


def test_hibrido_expoe_as_quatro_estrategias(hybrid_service):
    strategies = hybrid_service.strategies

    assert set(strategies) == set(Strategy)
    assert strategies[Strategy.cf] is hybrid_service.model.cf


def test_sem_hibrido_so_popularidade(popular_service):
    assert set(popular_service.strategies) == {Strategy.popular}
    with pytest.raises(ModelUnavailableError, match="POST /model/retrain"):
        popular_service.model_for(Strategy.hybrid)


def test_estrategia_padrao_e_a_do_modelo(hybrid_service):
    chosen, model = hybrid_service.model_for(None)

    assert chosen == Strategy.hybrid
    assert model is hybrid_service.model


@pytest.mark.parametrize(
    ("strategy", "reason_start"),
    [
        (Strategy.cf, "Porque você jogou"),
        (Strategy.content, "Parecido com"),
        (Strategy.popular, "Popular entre jogadores"),
    ],
)
def test_cada_estrategia_usa_seu_modelo(
    hybrid_service, seeded_db, db_session, strategy, reason_start
):
    # Usuário 7 (TF2, Portal 2, Half-Life 2): o CF encontra vizinhos de TF2 (4 e 6).
    out = hybrid_service.recommend(db_session, 7, k=2, strategy=strategy)

    assert out.strategy == strategy
    assert out.recommendations
    assert all(r.reason.startswith(reason_start) for r in out.recommendations)


def test_exclude_owned(hybrid_service, seeded_db, db_session):
    owned = {1, 2, 3}
    with_owned = hybrid_service.recommend(db_session, 1, k=9, exclude_owned=False)
    without = hybrid_service.recommend(db_session, 1, k=9)

    assert owned & {r.game_id for r in with_owned.recommendations}
    assert not owned & {r.game_id for r in without.recommendations}


def test_reason_aparece_nos_similares(hybrid_service, seeded_db, db_session):
    out = hybrid_service.similar_games(db_session, 1, k=2)

    assert all(r.reason for r in out.similar)


def test_api_responde_503_e_422(client, seeded_db):
    unavailable = client.get("/users/1/recommendations", params={"strategy": "hybrid"})
    invalid = client.get("/users/1/recommendations", params={"strategy": "aleatoria"})

    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "model_unavailable"
    assert invalid.status_code == 422
