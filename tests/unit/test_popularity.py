import joblib
import pandas as pd
import pytest

from recsys.base import Recommender
from recsys.models.popularity import POPULAR_REASON, PopularityRecommender


@pytest.fixture
def interactions() -> pd.DataFrame:
    # Jogo 10: 3 usuários; jogo 20: 2 usuários (muito jogado); jogo 30: 1 usuário.
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 3],
            "game_id": [10, 10, 10, 20, 20, 30],
            "score": [1.0, 1.0, 1.0, 9.0, 9.0, 1.0],
        }
    )


@pytest.fixture
def games() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [10, 20, 30, 40], "name": ["A", "B", "C", "D"]})


def test_segue_o_contrato():
    assert isinstance(PopularityRecommender(), Recommender)


def test_ordena_por_numero_de_usuarios(interactions, games):
    model = PopularityRecommender()
    model.fit(interactions, games)

    recs = model.recommend({}, k=4)

    assert [r.game_id for r in recs] == [10, 20, 30, 40]
    assert recs[0].score == pytest.approx(1.0)
    assert recs[1].score == pytest.approx(2 / 3)
    assert recs[3].score == 0.0
    assert all(r.reason == POPULAR_REASON for r in recs)


def test_ordena_por_score_acumulado(interactions):
    model = PopularityRecommender(by="score")
    model.fit(interactions)

    assert [r.game_id for r in model.recommend({}, k=3)] == [20, 10, 30]


def test_respeita_exclude_disliked_e_k(interactions, games):
    model = PopularityRecommender()
    model.fit(interactions, games)

    recs = model.recommend({10: 2.0}, k=1, exclude={10}, disliked={20})

    assert [r.game_id for r in recs] == [30]


def test_historico_nao_e_excluido_sem_exclude(interactions):
    model = PopularityRecommender()
    model.fit(interactions)

    assert model.recommend({10: 1.0}, k=1)[0].game_id == 10


def test_similar_games_exclui_o_proprio_jogo(interactions):
    model = PopularityRecommender()
    model.fit(interactions)

    assert [r.game_id for r in model.similar_games(10, k=2)] == [20, 30]


def test_add_game_entra_no_fim_do_ranking(interactions):
    model = PopularityRecommender()
    model.fit(interactions)
    model.add_game(99, "Novo Jogo")
    model.add_game(99, "Novo Jogo")

    ids = [r.game_id for r in model.recommend({}, k=10)]

    assert ids == [10, 20, 30, 99]


def test_save_e_load(tmp_path, interactions):
    model = PopularityRecommender()
    model.fit(interactions)
    path = tmp_path / "sub" / "pop.joblib"

    model.save(str(path))
    loaded = PopularityRecommender.load(str(path))

    assert loaded.recommend({}, k=3) == model.recommend({}, k=3)


def test_load_com_tipo_errado(tmp_path):
    path = tmp_path / "x.joblib"
    joblib.dump({"nao": "modelo"}, path)
    with pytest.raises(TypeError):
        PopularityRecommender.load(str(path))


def test_by_invalido():
    with pytest.raises(ValueError):
        PopularityRecommender(by="horas")


def test_fit_vazio():
    model = PopularityRecommender()
    model.fit(pd.DataFrame(columns=["user_id", "game_id", "score"]))

    assert model.recommend({}, k=5) == []
