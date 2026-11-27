import numpy as np
import pandas as pd
import pytest

from recsys import data
from recsys.data import (
    build_interactions,
    build_matrix,
    clean_games,
    clean_plays,
    clean_purchases,
    confidence_score,
    game_id_aliases,
    load_dataset,
)


@pytest.fixture
def raw_dir(tmp_path):
    """Dataset mínimo no formato original (TSV com cabeçalho).

    O jogo 5 ("Fallout 4 ") repete o nome do jogo 1 com um espaço no fim.
    """
    (tmp_path / "game_purchase.dat").write_text(
        "User_ID\tGame_ID\tPurchase\n"
        "1\t1\t1.0\n1\t2\t1.0\n1\t2\t1.0\n2\t1\t1.0\n2\t3\t1.0\n2\t5\t1.0\n"
    )
    (tmp_path / "game_play.dat").write_text(
        "User_ID\tGame_ID\tHours\n1\t1\t10.0\n2\t1\t1.5\n2\t1\t0.5\n2\t5\t1.0\n"
    )
    (tmp_path / "item_info.dat").write_text(
        "Game_ID\tGame Name\n1\tFallout 4\n2\t Dota 2 \n3\tPortal 2\n4\tSem Jogadores\n"
        "5\tFallout 4 \n"
    )
    (tmp_path / "user_info.dat").write_text("New_ID\tReal_ID\n1\t151603712\n2\t187131847\n")
    return tmp_path


def test_confidence_score():
    assert confidence_score(0.0) == 1.0
    assert confidence_score(np.e - 1, alpha=2.0) == pytest.approx(3.0)
    np.testing.assert_allclose(confidence_score(np.array([0.0, -5.0])), [1.0, 1.0])
    with pytest.raises(ValueError):
        confidence_score(1.0, alpha=-1)


def test_clean_purchases_remove_duplicados():
    raw = pd.DataFrame({"User_ID": [1, 1, 2], "Game_ID": [5, 5, 5], "Purchase": [1.0] * 3})

    assert clean_purchases(raw).to_dict("records") == [
        {"user_id": 1, "game_id": 5},
        {"user_id": 2, "game_id": 5},
    ]


def test_clean_plays_soma_horas_duplicadas_e_trata_negativas():
    raw = pd.DataFrame({"User_ID": [1, 1, 2], "Game_ID": [5, 5, 5], "Hours": [1.9, 0.2, -3]})

    out = clean_plays(raw).set_index("user_id")["hours"]

    assert out[1] == pytest.approx(2.1)
    assert out[2] == 0.0


def test_game_id_aliases_unifica_nomes_repetidos():
    raw = pd.DataFrame({"Game_ID": [7, 3, 9], "Game Name": ["Portal ", "Portal", "Dota 2"]})

    assert game_id_aliases(raw) == {7: 3}


def test_clean_purchases_e_plays_aplicam_aliases():
    purchases = pd.DataFrame({"User_ID": [1, 1], "Game_ID": [3, 7], "Purchase": [1.0, 1.0]})
    plays = pd.DataFrame({"User_ID": [1, 1], "Game_ID": [3, 7], "Hours": [1.0, 2.0]})

    assert clean_purchases(purchases, {7: 3}).to_dict("records") == [{"user_id": 1, "game_id": 3}]
    assert clean_plays(plays, {7: 3})["hours"].tolist() == [3.0]


def test_clean_games_tira_espacos_e_cria_tags():
    raw = pd.DataFrame({"Game_ID": [2, 1], "Game Name": [" Dota 2 ", "Fallout 4"]})

    out = clean_games(raw)

    assert out["game_id"].tolist() == [1, 2]
    assert out["name"].tolist() == ["Fallout 4", "Dota 2"]
    assert out["tags"].tolist() == [[], []]


def test_clean_games_mantem_menor_id_para_nome_repetido():
    raw = pd.DataFrame({"Game_ID": [7, 3], "Game Name": ["Portal ", "Portal"]})

    assert clean_games(raw)[["game_id", "name"]].to_dict("records") == [
        {"game_id": 3, "name": "Portal"}
    ]


def test_build_interactions_une_compras_e_horas():
    purchases = pd.DataFrame({"user_id": [1, 1], "game_id": [1, 2]})
    plays = pd.DataFrame({"user_id": [1, 3], "game_id": [1, 9], "hours": [10.0, 1.0]})

    out = build_interactions(purchases, plays, alpha=1.0).set_index(["user_id", "game_id"])

    assert len(out) == 3
    assert out.loc[(1, 1), "score"] == pytest.approx(1 + np.log1p(10.0))
    assert out.loc[(1, 2), "hours"] == 0.0
    assert out.loc[(1, 2), "score"] == 1.0
    assert out.loc[(3, 9), "score"] == pytest.approx(1 + np.log1p(1.0))


def test_load_dataset(raw_dir):
    ds = load_dataset(raw_dir, alpha=0.5)

    inter = ds.interactions.set_index(["user_id", "game_id"])
    assert len(inter) == 4
    # Jogo 5 unificado com o 1: horas somadas (1.5 + 0.5 + 1.0), sem par repetido.
    assert inter.loc[(2, 1), "hours"] == pytest.approx(3.0)
    assert inter.loc[(2, 1), "score"] == pytest.approx(1 + 0.5 * np.log1p(3.0))
    assert 5 not in ds.interactions["game_id"].tolist()
    assert ds.games["name"].tolist() == ["Fallout 4", "Dota 2", "Portal 2", "Sem Jogadores"]
    assert ds.users.columns.tolist() == ["user_id", "steam_real_id"]


def test_load_dataset_sem_arquivos(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        load_dataset(tmp_path)


def test_build_matrix_com_ids_padrao(raw_dir):
    ds = load_dataset(raw_dir)

    m = build_matrix(ds.interactions)

    assert m.shape == (2, 3)
    assert m.matrix.dtype == np.float32
    assert m.matrix[m.user_index[1], m.game_index[2]] == pytest.approx(1.0)
    assert m.matrix[m.user_index[2], m.game_index[2]] == 0.0
    assert m.sparsity == pytest.approx(2 / 6)


def test_build_matrix_com_catalogo_completo(raw_dir):
    ds = load_dataset(raw_dir)

    m = build_matrix(ds.interactions, game_ids=ds.games["game_id"].to_numpy())

    assert m.shape == (2, 4)
    assert m.matrix[:, m.game_index[4]].nnz == 0


def test_build_matrix_ignora_ids_desconhecidos():
    inter = pd.DataFrame({"user_id": [1, 2], "game_id": [1, 1], "score": [2.0, 3.0]})

    m = build_matrix(inter, user_ids=np.array([1]))

    assert m.shape == (1, 1)
    assert m.matrix.sum() == pytest.approx(2.0)


def test_sparsity_de_matriz_vazia():
    m = build_matrix(pd.DataFrame(columns=["user_id", "game_id", "score"]))

    assert m.sparsity == 0.0


@pytest.mark.skipif(
    not (data.DEFAULT_DATA_DIR / data.PURCHASE_FILE).exists(), reason="dataset real ausente"
)
def test_dataset_real_tem_os_numeros_da_analise_exploratoria():
    ds = load_dataset()

    # 5.155 jogos no original; 2 repetem o nome de outro e são unificados.
    assert len(ds.games) == 5_153
    assert ds.games["name"].is_unique
    assert len(ds.interactions) == 128_801
    assert ds.interactions["user_id"].nunique() == 12_393
