import json

import pytest
from sqlalchemy import select

from app.db.models import Game, ModelRun
from recsys.models.hybrid import HybridRecommender
from recsys.train import (
    ARTIFACT_NAME,
    DEFAULT_PARAMS,
    build_hybrid,
    load_from_db,
    main,
    register_run,
    save_atomically,
    search_hyperparameters,
    train,
    with_scores,
)


@pytest.fixture
def data(synthetic_dataset):
    return synthetic_dataset.interactions[["user_id", "game_id", "hours"]], synthetic_dataset.games


def test_with_scores_recalcula_pelo_alpha(data):
    scored = with_scores(data[0], alpha=0.0)

    assert set(scored["score"]) == {1.0}
    assert "score" not in data[0]


def test_build_hybrid_usa_os_parametros():
    model = build_hybrid({**DEFAULT_PARAMS, "k_neighbors": 7, "dislike_penalty": 0.2})

    assert model.cf.k_neighbors == 7
    assert model.dislike_penalty == 0.2
    assert model.weights_for(0) == (0.0, 0.0, 1.0)


def test_busca_escolhe_a_maior_ndcg(data):
    best, table = search_hyperparameters(
        *data, grid={"k_neighbors": [1, 5]}, k=3, max_users=None, min_interactions=3
    )

    assert [row["k_neighbors"] for row in table] == [1, 5]
    top = max(table, key=lambda row: row["ndcg@3"])
    assert best["k_neighbors"] == top["k_neighbors"]
    assert best["alpha"] == DEFAULT_PARAMS["alpha"]


def test_train_grava_artefato_com_metadados(tmp_path, data):
    result = train(*data, artifacts_dir=tmp_path, search=False, k=3, min_interactions=3)

    path = tmp_path / ARTIFACT_NAME
    assert result.artifact_path == str(path)
    assert not (tmp_path / (ARTIFACT_NAME + ".tmp")).exists()
    loaded = HybridRecommender.load(str(path))
    assert loaded.metadata_["params"] == result.params
    assert set(result.metrics) >= {"ndcg@3", "coverage"}
    json.dumps(result.params)  # vai para uma coluna JSON
    # treinado com todos os dados: o usuário novo recebe os populares
    assert [r.game_id for r in loaded.recommend({}, k=1)] == [2]


def test_save_atomically_sobrescreve(tmp_path, data):
    model = build_hybrid(DEFAULT_PARAMS)
    model.fit(with_scores(data[0], 1.0), data[1])
    path = tmp_path / "sub" / "m.joblib"

    save_atomically(model, path)
    save_atomically(model, path)

    assert [p.name for p in path.parent.iterdir()] == ["m.joblib"]


def test_load_from_db_e_register_run(seeded_db, engine, db_session, tmp_path):
    interactions, games = load_from_db(engine)
    assert len(interactions) == len(seeded_db.interactions)
    assert len(games) == 9

    result = train(interactions, games, artifacts_dir=tmp_path, search=False, min_interactions=3)
    run_id = register_run(engine, result)

    run = db_session.scalar(select(ModelRun).where(ModelRun.id == run_id))
    assert run.metrics == result.metrics
    assert run.artifact_path.endswith(ARTIFACT_NAME)


def test_main_com_banco(seeded_db, engine, db_session, tmp_path, capsys):
    db_session.add(Game(name="Jogo Sem Interações"))
    db_session.commit()

    result = main(["--artifacts-dir", str(tmp_path), "--no-search", "--k", "3"], engine=engine)

    out = capsys.readouterr().out
    assert "registrado em model_runs" in out
    assert (tmp_path / ARTIFACT_NAME).exists()
    assert 10 in result.model.content.game_index_


def test_main_com_arquivos_e_busca(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "game_purchase.dat").write_text(
        "User_ID\tGame_ID\tPurchase\n"
        + "".join(f"{u}\t{g}\t1.0\n" for u in range(1, 5) for g in range(1, 8))
    )
    (raw / "game_play.dat").write_text("User_ID\tGame_ID\tHours\n1\t1\t5.0\n")
    (raw / "item_info.dat").write_text(
        "Game_ID\tGame Name\n" + "".join(f"{g}\tJogo {g}\n" for g in range(1, 8))
    )
    (raw / "user_info.dat").write_text("New_ID\tReal_ID\n1\t10\n2\t20\n3\t30\n4\t40\n")

    result = main(
        ["--source", "files", "--data-dir", str(raw), "--artifacts-dir", str(tmp_path / "art")]
    )

    out = capsys.readouterr().out
    assert "busca:" in out
    assert "registrado" not in out
    assert result.params["k_neighbors"] in {20, 50, 100}
