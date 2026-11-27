import pytest

from recsys.base import Recommendation, Recommender
from recsys.models.hybrid import DEFAULT_WEIGHTS, HybridRecommender, normalize_scores
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import POPULAR_REASON


@pytest.fixture
def model(synthetic_dataset):
    m = HybridRecommender(cf=ItemKNNRecommender(k_neighbors=5))
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)
    return m


def ids(recs):
    return [r.game_id for r in recs]


def test_segue_o_contrato(model):
    assert isinstance(model, Recommender)


def test_normalize_scores():
    recs = [Recommendation(1, 4.0), Recommendation(2, 2.0), Recommendation(3, 1.0)]

    assert normalize_scores(recs) == {1: 1.0, 2: pytest.approx(1 / 3), 3: 0.0}
    assert normalize_scores([Recommendation(1, 5.0), Recommendation(2, 5.0)]) == {1: 1.0, 2: 1.0}
    assert normalize_scores([]) == {}


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, (0, 0, 1)), (1, (0.3, 0.3, 0.4)), (2, (0.3, 0.3, 0.4)), (9, (0.7, 0.2, 0.1))],
)
def test_pesos_pelo_tamanho_do_historico(model, size, expected):
    assert model.weights_for(size) == expected


def test_usuario_novo_recebe_popularidade(model):
    recs = model.recommend({}, k=3)

    assert ids(recs) == [2, 5, 6]
    assert all(r.reason == POPULAR_REASON for r in recs)


def test_com_historico_usa_cf_e_conteudo_com_motivo(model):
    recs = model.recommend({1: 3.0, 2: 2.0, 3: 2.0}, k=2, exclude={1, 2, 3})

    # Fallout 3 (9) não tem interações: só o conteúdo (franquia) o traz para o topo.
    assert 9 in ids(recs)
    reasons = {r.game_id: r.reason for r in recs}
    assert reasons[9].startswith("Parecido com")


def test_historico_curto_mistura_os_tres(model):
    recs = model.recommend({4: 5.0}, k=3, exclude={4})

    assert ids(recs)[0] == 6  # vizinho de Dota 2 no CF e popular
    assert all(0 < r.score <= 1 for r in recs)


def test_exclude_e_disliked_nunca_aparecem(model):
    recs = model.recommend({1: 1.0}, k=9, exclude={1}, disliked={2})

    assert not {1, 2} & set(ids(recs))


def test_nao_gostei_penaliza_jogos_parecidos(model):
    base = model.recommend({}, k=9)
    penalized = model.recommend({}, k=9, disliked={4})

    # Counter-Strike (6) é vizinho de Dota 2 (4) e cai no ranking.
    assert ids(base).index(6) < ids(penalized).index(6)


def test_sem_penalidade_a_ordem_nao_muda(synthetic_dataset):
    m = HybridRecommender(dislike_penalty=0)
    m.fit(synthetic_dataset.interactions, synthetic_dataset.games)

    expected = [g for g in ids(m.recommend({}, k=9)) if g != 4]
    assert ids(m.recommend({}, k=9, disliked={4})) == expected


def test_similar_games_combina_cf_e_conteudo(model):
    similar = model.similar_games(1, k=3)

    assert set(ids(similar)) == {2, 3, 9}
    assert model.similar_games(999) == []


def test_add_game_repassa_aos_componentes(model):
    model.add_game(50, "Fallout 76", ["rpg"])

    assert 50 in model.content.game_index_
    assert 50 in model.cf.game_index_
    assert 50 in ids(model.similar_games(1, k=5))
    assert 50 in ids(model.recommend({1: 1.0, 2: 1.0, 3: 1.0}, k=5, exclude={1, 2, 3}))


def test_save_e_load(tmp_path, model):
    model.metadata_ = {"trained_at": "2026-11-26T10:00:00+00:00"}
    path = tmp_path / "hybrid.joblib"
    model.save(str(path))

    loaded = HybridRecommender.load(str(path))

    assert loaded.metadata_["trained_at"].startswith("2026-11-26")
    assert loaded.recommend({1: 1.0}, k=3) == model.recommend({1: 1.0}, k=3)


@pytest.mark.parametrize(
    "kwargs", [{"weights": ((1, (1, 0, 0)),)}, {"weights": ()}, {"dislike_penalty": -1}]
)
def test_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        HybridRecommender(**kwargs)


def test_pesos_padrao_por_faixa_de_historico():
    assert DEFAULT_WEIGHTS[0] == (0, (0.0, 0.0, 1.0))
