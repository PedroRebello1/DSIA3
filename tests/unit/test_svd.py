import numpy as np
import pandas as pd
import pytest

from recsys.base import Recommender
from recsys.models.svd import SVDRecommender

RPG, SHOOTERS = {1, 2, 3}, {4, 5, 6}


@pytest.fixture
def model(synthetic_dataset):
    m = SVDRecommender(n_factors=3)
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    return m


def test_segue_o_contrato(model):
    assert isinstance(model, Recommender)


def test_fatores_por_jogo(model):
    assert model.item_factors_.shape == (9, 3)
    assert 0 < model.explained_variance_ratio_ <= 1
    np.testing.assert_array_equal(model.item_factors_[model.game_index_[9]], 0)


def test_n_factors_limitado_pelo_tamanho_da_matriz(synthetic_dataset):
    m = SVDRecommender(n_factors=500)
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)

    assert m.item_factors_.shape[1] == 6  # min(7 usuários, 9 jogos) - 1


def test_recomenda_o_grupo_do_historico(model):
    recs = model.recommend({1: 3.0, 2: 2.0}, k=1, exclude={1, 2})

    assert recs[0].game_id == 3
    assert recs[0].reason in {"Porque você jogou Fallout 4", "Porque você jogou Fallout New Vegas"}


def test_scores_sao_h_v_vt(model):
    factors = model.item_factors_
    h = np.zeros(9)
    h[model.game_index_[4]] = 2.0
    expected = h @ factors @ factors.T

    recs = model.recommend({4: 2.0}, k=3, exclude={4})

    for r in recs:
        assert r.score == pytest.approx(expected[model.game_index_[r.game_id]], rel=1e-4)
    assert {r.game_id for r in recs[:2]} == {5, 6}


def test_exclude_disliked_e_historico_vazio(model):
    recs = model.recommend({1: 1.0}, k=9, exclude={1}, disliked={2, 3})
    assert not {1, 2, 3} & {r.game_id for r in recs}
    assert model.recommend({}, k=3) == []
    assert model.recommend({5: 1.0}, k=3, disliked={5}) == []


def test_similar_games(model):
    similar = model.similar_games(4, k=2)

    assert {r.game_id for r in similar} <= SHOOTERS - {4} | {7, 8}
    assert similar[0].reason == "Quem jogou Dota 2 também jogou"
    assert model.similar_games(999) == []


def test_add_game(model):
    model.add_game(50, "Jogo Novo")
    model.add_game(50, "Jogo Novo")

    assert model.item_factors_.shape == (10, 3)
    assert model.similar_games(50) == []
    assert 50 not in {r.game_id for r in model.recommend({50: 1.0, 1: 1.0}, k=10)}


def test_save_e_load(tmp_path, model):
    path = tmp_path / "svd.joblib"
    model.save(str(path))

    assert SVDRecommender.load(str(path)).recommend({1: 1.0}, k=3) == model.recommend({1: 1.0}, k=3)


def test_matriz_pequena_demais():
    m = SVDRecommender()
    m.fit(pd.DataFrame({"user_id": [1], "game_id": [1], "score": [1.0]}))

    assert m.recommend({1: 1.0}) == []


def test_n_factors_invalido():
    with pytest.raises(ValueError):
        SVDRecommender(n_factors=0)
