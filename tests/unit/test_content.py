import numpy as np
import pandas as pd
import pytest

from recsys.base import Recommender
from recsys.models.content import ContentRecommender


@pytest.fixture
def model(synthetic_dataset):
    m = ContentRecommender()
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    return m


def test_segue_o_contrato(model):
    assert isinstance(model, Recommender)


def test_vetores_tem_norma_um(model):
    norms = np.sqrt(np.asarray(model.vectors_.multiply(model.vectors_).sum(axis=1))).ravel()
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_similar_games_encontra_a_franquia(model):
    similar = model.similar_games(1, k=2)

    assert {r.game_id for r in similar} == {2, 9}  # Fallout New Vegas e Fallout 3
    assert similar[0].reason == "Parecido com Fallout 4"
    assert model.similar_games(999) == []


def test_tags_aproximam_jogos_sem_nome_parecido(model):
    shooters = {r.game_id for r in model.similar_games(6, k=2)}  # Counter-Strike

    assert shooters == {5, 8}  # também têm a tag "tiro"


def test_recomenda_jogo_sem_interacoes(model):
    recs = model.recommend({2: 3.0}, k=2, exclude={2})

    assert recs[0].game_id in {1, 9}
    assert {r.game_id for r in recs} == {1, 9}
    assert recs[0].reason == "Parecido com Fallout New Vegas, que você jogou"


def test_exclude_disliked_e_historico_vazio(model):
    recs = model.recommend({2: 1.0}, k=9, exclude={2}, disliked={1, 9})

    assert not {1, 2, 9} & {r.game_id for r in recs}
    assert model.recommend({}, k=3) == []
    assert model.recommend({2: 1.0}, k=3, disliked={2}) == []


def test_add_game_fica_recomendavel_na_hora(model):
    model.add_game(50, "Fallout 76", ["rpg", "online"])

    assert model.similar_games(50, k=1)[0].game_id in {1, 2, 9}
    assert 50 in {r.game_id for r in model.recommend({1: 1.0}, k=3, exclude={1})}


def test_add_game_existente_atualiza_o_vetor(model):
    before = model.vectors_.shape
    model.add_game(7, "Portal 2", ["tiro"])

    assert model.vectors_.shape == before
    assert 7 in {r.game_id for r in model.similar_games(5, k=3)}


def test_sem_tags_no_catalogo():
    games = pd.DataFrame(
        {"game_id": [1, 2, 3], "name": ["Total War Rome", "Total War Shogun", "Doom"]}
    )
    m = ContentRecommender()
    m.fit(None, games)
    assert m.similar_games(1, k=1)[0].game_id == 2

    # Tags que não existiam no treino valem na hora (hashing, sem vocabulário fixo).
    m.add_game(4, "Alpha", ["estratégia", "medieval"])
    m.add_game(5, "Beta", ["estratégia", "medieval"])
    assert m.similar_games(4, k=1)[0].game_id == 5


def test_fit_sem_catalogo():
    with pytest.raises(ValueError):
        ContentRecommender().fit(None, pd.DataFrame())


def test_save_e_load(tmp_path, model):
    path = tmp_path / "content.joblib"
    model.save(str(path))
    loaded = ContentRecommender.load(str(path))
    loaded.add_game(60, "Half-Life 3", ["tiro"])

    assert loaded.recommend({1: 1.0}, k=3) == model.recommend({1: 1.0}, k=3)
    assert loaded.similar_games(60, k=1)[0].game_id == 8
