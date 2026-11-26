import pytest

from recsys import tuning
from recsys.evaluation import Split, train_test_split
from recsys.train import DEFAULT_PARAMS
from recsys.tuning import run_experiment, truncate_histories, tune, with_band


def test_with_band_troca_so_a_faixa_pedida():
    params = with_band(DEFAULT_PARAMS, 1, (0.5, 0.25, 0.25))

    bands = dict((size, tuple(w)) for size, w in params["weights"])
    assert bands[1] == (0.5, 0.25, 0.25)
    assert bands[3] == tuple(dict(DEFAULT_PARAMS["weights"])[3])
    assert dict(DEFAULT_PARAMS["weights"])[1] != [0.5, 0.25, 0.25]  # original intacto


def test_truncate_histories_limita_so_usuarios_avaliados(synthetic_dataset):
    split = train_test_split(synthetic_dataset.interactions, min_interactions=3)
    truncated = truncate_histories(split, limit=1)

    sizes = truncated.train.groupby("user_id").size()
    assert all(sizes[u] == 1 for u in split.test)
    others = set(split.train.user_id) - set(split.test)
    assert all(sizes[u] == (split.train.user_id == u).sum() for u in others)
    assert isinstance(truncated, Split)


@pytest.fixture
def hours(synthetic_dataset):
    return synthetic_dataset.interactions[["user_id", "game_id", "hours"]], synthetic_dataset.games


def test_run_experiment_com_e_sem_truncar(hours):
    params = {**DEFAULT_PARAMS, "k_neighbors": 5}

    full = run_experiment(*hours, params, min_interactions=3)
    short = run_experiment(*hours, params, history_limit=1, min_interactions=3)

    assert full["users"] == short["users"] > 0
    assert set(full) >= {"ndcg@10", "coverage"}


def test_tune_escolhe_o_melhor_de_cada_etapa(hours):
    params, tables = tune(
        *hours,
        base_params={"k_neighbors": 5},
        max_users=None,
        alphas=[0.5, 1.0],
        weights_long=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        weights_short=[(0.5, 0.5, 0.0)],
        min_interactions=3,
    )

    assert list(tables) == ["alpha", "weights_long", "weights_short"]
    best_alpha = tables["alpha"]["ndcg@10"].idxmax()
    assert f"α = {params['alpha']:g}" == best_alpha
    assert dict(params["weights"])[1] == [0.5, 0.5, 0.0]


def test_main_imprime_as_tabelas(monkeypatch, capsys, synthetic_dataset):
    monkeypatch.setattr(tuning, "load_dataset", lambda: synthetic_dataset)
    original = tuning.tune

    def small_tune(*args, **kwargs):
        return original(
            *args,
            **kwargs,
            alphas=[1.0],
            weights_long=[(0.7, 0.2, 0.1)],
            weights_short=[(0.3, 0.3, 0.4)],
            min_interactions=3,
        )

    monkeypatch.setattr(tuning, "tune", small_tune)
    tuning.main(["--max-users", "5", "--k-neighbors", "5"])

    out = capsys.readouterr().out
    assert "### α do score de confiança" in out
    assert "Parâmetros escolhidos" in out
