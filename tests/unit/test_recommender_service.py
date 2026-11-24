from datetime import UTC, datetime

import pytest

from app.core.errors import NotFoundError
from app.db import crud
from app.services.recommender_service import (
    RecommenderService,
    build_profile,
    load_training_data,
)
from recsys.models.popularity import PopularityRecommender


@pytest.fixture
def service(seeded_db, db_session):
    return RecommenderService.from_db(db_session)


def test_load_training_data_ignora_nao_gostei(seeded_db, db_session):
    crud.upsert_interaction(db_session, 1, 1, disliked=True)
    db_session.commit()

    interactions, games = load_training_data(db_session)

    assert len(interactions) == len(seeded_db.interactions) - 1
    assert interactions["score"].min() == 1.0
    assert len(games) == 9
    assert set(games.columns) == {"game_id", "name", "tags"}


def test_build_profile_separa_positivos_e_nao_gostei(seeded_db, db_session):
    crud.upsert_interaction(db_session, 2, 1, disliked=True)
    profile = build_profile(crud.get_interactions(db_session, 2))

    assert profile.owned == {1, 2}
    assert profile.disliked == {1}
    assert list(profile.history) == [2]
    assert profile.history[2] > 1.0  # 30 horas jogadas


def test_from_db_treina_popularidade(service):
    assert isinstance(service.model, PopularityRecommender)
    assert service.strategy == "popular"
    assert datetime.fromisoformat(service.version) <= datetime.now(UTC)


def test_recomenda_sem_jogos_do_historico(service, db_session):
    out = service.recommend(db_session, 1, k=3)

    # Usuário 1 tem os jogos 1, 2 e 3; os mais populares a seguir são 5 e 6 (3 usuários) e 4.
    assert [r.game_id for r in out.recommendations] == [5, 6, 4]
    assert out.recommendations[0].name == "Team Fortress 2"
    assert out.user_id == 1
    assert out.model_version == service.version


def test_nao_gostei_tambem_e_excluido(service, db_session):
    crud.upsert_interaction(db_session, 1, 5, disliked=True)

    ids = [r.game_id for r in service.recommend(db_session, 1, k=3).recommendations]

    assert 5 not in ids


def test_usuario_inexistente(service, db_session):
    with pytest.raises(NotFoundError):
        service.recommend(db_session, 99, k=3)
