import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from recsys.base import Recommender
from recsys.data import build_matrix
from recsys.models.item_knn import ItemKNNRecommender

RPG = {1, 2, 3}


@pytest.fixture
def model(synthetic_dataset):
    m = ItemKNNRecommender(k_neighbors=5)
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    return m


def test_segue_o_contrato(model):
    assert isinstance(model, Recommender)


def test_similaridade_e_o_cosseno_entre_colunas(model, synthetic_dataset):
    matrix = build_matrix(synthetic_dataset.interactions, game_ids=model.game_ids_).matrix
    expected = cosine_similarity(matrix.T.toarray())
    i, j = model.game_index_[1], model.game_index_[2]

    assert model.similarity_[i, j] == pytest.approx(expected[i, j], rel=1e-5)
    assert model.similarity_.diagonal().sum() == 0


def test_guarda_no_maximo_k_vizinhos(synthetic_dataset):
    m = ItemKNNRecommender(k_neighbors=1)
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)

    assert np.diff(m.similarity_.indptr).max() == 1


def test_jogo_sem_interacao_fica_no_catalogo_sem_vizinhos(model):
    assert model.similar_games(9) == []
    assert 9 in model.game_index_


def test_recomenda_jogos_do_mesmo_grupo_com_motivo(model):
    recs = model.recommend({1: 3.0}, k=5, exclude={1})

    assert {r.game_id for r in recs} == {2, 3}
    assert all(r.reason == "Porque você jogou Fallout 4" for r in recs)
    assert recs[0].score >= recs[1].score > 0


def test_motivo_aponta_o_jogo_que_mais_contribuiu(model):
    recs = model.recommend({1: 1.0, 5: 5.0}, k=10, exclude={1, 5})
    reasons = {r.game_id: r.reason for r in recs}

    assert reasons[2] == "Porque você jogou Fallout 4"
    assert reasons[7] == "Porque você jogou Team Fortress 2"


def test_exclude_e_disliked(model):
    recs = model.recommend({1: 1.0, 2: 1.0}, k=10, exclude={1, 2}, disliked={3})
    assert recs == []

    # Jogo marcado como "não gostei" não entra como histórico positivo.
    only_disliked = model.recommend({4: 5.0}, k=10, disliked={4})
    assert only_disliked == []


def test_historico_vazio_ou_desconhecido(model):
    assert model.recommend({}, k=5) == []
    assert model.recommend({999: 1.0}, k=5) == []


def test_similar_games(model):
    similar = model.similar_games(4, k=2)

    assert [r.game_id for r in similar][0] == 6
    assert all(r.game_id not in RPG for r in similar)
    assert similar[0].reason == "Quem jogou Dota 2 também jogou"
    assert model.similar_games(999) == []


def test_add_game(model):
    model.add_game(50, "Jogo Novo")
    model.add_game(50, "Jogo Novo (renomeado)")

    assert model.similarity_.shape == (10, 10)
    assert model.similar_games(50) == []
    assert model.names_[50] == "Jogo Novo (renomeado)"
    assert {r.game_id for r in model.recommend({50: 1.0, 1: 1.0}, k=5, exclude={1})} == {2, 3}


def test_save_e_load(tmp_path, model):
    path = tmp_path / "knn.joblib"
    model.save(str(path))

    loaded = ItemKNNRecommender.load(str(path))

    assert loaded.recommend({1: 1.0}, k=3) == model.recommend({1: 1.0}, k=3)


def test_fit_sem_catalogo_e_em_blocos():
    inter = pd.DataFrame(
        {"user_id": [1, 1, 2, 2, 3], "game_id": [10, 20, 10, 20, 30], "score": [1.0] * 5}
    )
    m = ItemKNNRecommender(k_neighbors=2, chunk_size=1)
    m.fit(inter)

    assert m.recommend({10: 1.0}, k=5, exclude={10})[0].game_id == 20
    assert m.recommend({10: 1.0}, k=5)[0].reason == "Porque você jogou o jogo 10"


def test_k_neighbors_invalido():
    with pytest.raises(ValueError):
        ItemKNNRecommender(k_neighbors=0)


def test_catalogo_com_um_jogo():
    m = ItemKNNRecommender()
    m.fit(pd.DataFrame({"user_id": [1], "game_id": [1], "score": [1.0]}))

    assert m.similarity_.nnz == 0
    assert m.recommend({1: 1.0}) == []
