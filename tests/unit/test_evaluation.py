import math

import pandas as pd
import pytest

from recsys.base import Recommendation
from recsys.evaluation import (
    Split,
    average_popularity,
    average_precision_at_k,
    catalog_coverage,
    compare_models,
    evaluate,
    hit_rate_at_k,
    main,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    to_markdown,
    train_test_split,
)
from recsys.models.popularity import PopularityRecommender

# Caso calculado à mão: acertos nas posições 2 e 4, três itens relevantes.
RECS = [1, 2, 3, 4]
RELEVANT = {2, 4, 9}


def test_metricas_caso_calculado_a_mao():
    assert precision_at_k(RECS, RELEVANT, 4) == 0.5  # 2 acertos / 4
    assert recall_at_k(RECS, RELEVANT, 4) == pytest.approx(2 / 3)
    assert hit_rate_at_k(RECS, RELEVANT, 4) == 1.0
    # AP = (1/2 + 2/4) / min(3, 4) = 1/3
    assert average_precision_at_k(RECS, RELEVANT, 4) == pytest.approx(1 / 3)
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(RECS, RELEVANT, 4) == pytest.approx(dcg / idcg)


def test_metricas_respeitam_o_corte_k():
    assert precision_at_k(RECS, RELEVANT, 1) == 0.0
    assert precision_at_k(RECS, RELEVANT, 2) == 0.5
    assert hit_rate_at_k(RECS, RELEVANT, 1) == 0.0
    assert ndcg_at_k([2], {2}, 10) == 1.0
    assert average_precision_at_k([9, 2, 4], RELEVANT, 3) == 1.0


def test_metricas_com_casos_vazios():
    assert precision_at_k([], RELEVANT, 0) == 0.0
    assert recall_at_k(RECS, set(), 4) == 0.0
    assert ndcg_at_k(RECS, set(), 4) == 0.0
    assert average_precision_at_k(RECS, set(), 4) == 0.0
    assert catalog_coverage([], 0) == 0.0
    assert novelty([], {}) == 0.0
    assert average_popularity([], {}) == 0.0


def test_cobertura_e_novidade():
    lists = [[1, 2], [2, 3]]
    popularity = {1: 0.5, 2: 0.25, 3: 0.0}

    assert catalog_coverage(lists, 10) == 0.3
    # -log2 de 0.5, 0.25, 0.25 e 0.25 (o jogo 3 usa a menor fração positiva)
    assert novelty(lists, popularity) == pytest.approx((1 + 2 + 2 + 2) / 4)
    assert average_popularity(lists, popularity) == pytest.approx((0.5 + 0.25 + 0.25 + 0) / 4)


@pytest.fixture
def interactions():
    # Usuário 1: 10 jogos; usuário 2: 5 jogos; usuário 3: 4 jogos (fica só no treino).
    rows = [(1, g) for g in range(10)] + [(2, g) for g in range(5)] + [(3, g) for g in range(4)]
    df = pd.DataFrame(rows, columns=["user_id", "game_id"])
    df["score"] = 1.0
    return df


def test_split_separa_vinte_por_cento_dos_usuarios_elegiveis(interactions):
    split = train_test_split(interactions, seed=7)

    assert set(split.test) == {1, 2}
    assert len(split.test[1]) == 2
    assert len(split.test[2]) == 1
    assert len(split.train) + 3 == len(interactions)
    train_pairs = set(zip(split.train.user_id, split.train.game_id, strict=True))
    assert not any((u, g) in train_pairs for u, games in split.test.items() for g in games)


def test_split_e_reprodutivel(interactions):
    a = train_test_split(interactions, seed=1)
    b = train_test_split(interactions, seed=1)

    assert a.test == b.test


def test_split_fracao_invalida(interactions):
    with pytest.raises(ValueError):
        train_test_split(interactions, test_fraction=1.0)


class FixedModel:
    """Recomenda sempre a mesma lista, menos o que for excluído."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def recommend(self, history, k=10, exclude=None, disliked=None):
        self.calls.append((dict(history), set(exclude or ())))
        ids = [g for g in self.ranking if g not in (exclude or set())][:k]
        return [Recommendation(g, 1.0) for g in ids]


def test_evaluate_com_modelo_fixo():
    train = pd.DataFrame({"user_id": [1, 1, 2], "game_id": [1, 2, 1], "score": [2.0, 1.0, 1.0]})
    split = Split(train=train, test={1: {3}, 2: {5}})
    model = FixedModel([1, 3, 4])

    result = evaluate(model, split, n_items=10, k=2)

    # Usuário 1 (exclui 1 e 2): recebe [3, 4] e acerta o 3. Usuário 2 (exclui 1): [3, 4], erra.
    assert model.calls[0] == ({1: 2.0, 2: 1.0}, {1, 2})
    assert result["precision@2"] == pytest.approx(0.25)
    assert result["recall@2"] == pytest.approx(0.5)
    assert result["hit_rate@2"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(0.2)
    assert result["users"] == 2


def test_evaluate_com_amostra_de_usuarios(interactions):
    split = train_test_split(interactions)

    result = evaluate(FixedModel([]), split, n_items=10, max_users=1)

    assert result["users"] == 1
    assert result["precision@10"] == 0.0


def test_compare_models_e_markdown(synthetic_dataset):
    table = compare_models({"pop": PopularityRecommender()}, synthetic_dataset, k=3)

    assert list(table.index) == ["pop"]
    assert {"ndcg@3", "coverage", "novelty", "fit_s"} <= set(table.columns)
    markdown = to_markdown(table)
    assert markdown.splitlines()[0].startswith("| modelo | precision@3")
    assert "| pop |" in markdown


def test_main_com_dataset_pequeno(tmp_path, capsys):
    (tmp_path / "game_purchase.dat").write_text(
        "User_ID\tGame_ID\tPurchase\n"
        + "".join(f"{u}\t{g}\t1.0\n" for u in range(1, 4) for g in range(1, 7))
    )
    (tmp_path / "game_play.dat").write_text("User_ID\tGame_ID\tHours\n1\t1\t5.0\n")
    (tmp_path / "item_info.dat").write_text(
        "Game_ID\tGame Name\n" + "".join(f"{g}\tJogo {g}\n" for g in range(1, 7))
    )
    (tmp_path / "user_info.dat").write_text("New_ID\tReal_ID\n1\t10\n2\t20\n3\t30\n")

    main(["--data-dir", str(tmp_path), "--k", "2"])

    out = capsys.readouterr().out
    assert "M1 item-kNN" in out
    assert "M3 conteúdo" in out
