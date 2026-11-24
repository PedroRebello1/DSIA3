import pytest
from pydantic import ValidationError

from app.core.errors import NotFoundError
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.services.preferences import replace_preferences, update_preferences
from app.services.users import get_user


def history(detail):
    return {h.game_id: (h.purchased, h.hours, h.disliked) for h in detail.history}


def test_patch_exige_alguma_alteracao_e_sem_conflito():
    with pytest.raises(ValidationError, match="ao menos um jogo"):
        PreferencesPatch()
    with pytest.raises(ValidationError, match=r"ao mesmo tempo: \[1\]"):
        PreferencesPatch(upsert=[{"game_id": 1}], remove=[1])
    with pytest.raises(ValidationError, match="jogos repetidos"):
        PreferencesPatch(upsert=[{"game_id": 1}, {"game_id": 1, "hours": 2}])
    with pytest.raises(ValidationError):
        PreferencesPatch(upsert=[{"game_id": 1, "hours": -3}])


def test_replace_troca_todo_o_historico(seeded_db, db_session):
    data = PreferencesReplace(games=[{"game_id": 7, "hours": 5}, {"game_id": 9, "disliked": True}])

    detail = replace_preferences(db_session, 1, data)

    assert history(detail) == {7: (True, 5.0, False), 9: (True, 0.0, True)}


def test_replace_com_lista_vazia_limpa_o_historico(seeded_db, db_session):
    assert replace_preferences(db_session, 2, PreferencesReplace(games=[])).history == []


def test_patch_atualiza_so_campos_informados_adiciona_e_remove(seeded_db, db_session):
    data = PreferencesPatch(
        upsert=[{"game_id": 1, "disliked": True}, {"game_id": 9, "hours": 7}],
        remove=[2, 8],  # 8 não está no histórico: ignorado
    )

    detail = update_preferences(db_session, 2, data)

    assert history(detail) == {1: (True, 50.0, True), 9: (True, 7.0, False)}


def test_erros_de_usuario_e_jogo(seeded_db, db_session):
    with pytest.raises(NotFoundError, match="Usuário 99"):
        update_preferences(db_session, 99, PreferencesPatch(remove=[1]))
    with pytest.raises(NotFoundError, match=r"\[42\]"):
        replace_preferences(db_session, 1, PreferencesReplace(games=[{"game_id": 42}]))
    # Nada foi alterado depois do erro.
    assert set(history(get_user(db_session, 1))) == {1, 2, 3}
