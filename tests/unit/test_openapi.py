import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.error import ErrorResponse, error_example
from app.schemas.game import GameCreate, GameOut
from app.schemas.health import HealthOut
from app.schemas.interaction import InteractionIn, InteractionOut
from app.schemas.model import ModelInfoOut, RetrainOut
from app.schemas.preference import PreferencesPatch, PreferencesReplace
from app.schemas.recommendation import RecommendationsOut, SimilarGamesOut
from app.schemas.user import UserCreate, UserDetail, UserOut


def spec():
    return TestClient(create_app()).get("/openapi.json").json()


def test_tags_documentadas():
    doc = spec()
    declared = {t["name"] for t in doc["tags"]}
    used = {tag for ops in doc["paths"].values() for op in ops.values() for tag in op["tags"]}

    assert used <= declared
    assert all(t["description"] for t in doc["tags"])
    assert "Fluxo típico" in doc["info"]["description"]


def test_toda_rota_tem_resumo_descricao_e_erros():
    for path, ops in spec()["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), f"{method} {path} sem summary"
            assert op.get("description"), f"{method} {path} sem descrição"
            if "{" in path or method in {"post", "put", "patch"}:
                assert "422" in op["responses"], f"{method} {path} sem 422"


def test_erros_usam_o_envelope_padrao():
    doc = spec()
    response = doc["paths"]["/users/{user_id}"]["get"]["responses"]["404"]

    ref = response["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/ErrorResponse")
    assert "error" in doc["components"]["schemas"]["ErrorResponse"]["properties"]


def resolve(doc, schema):
    """Schema do componente referenciado (direto ou como item de lista)."""
    ref = schema.get("$ref") or schema.get("items", {}).get("$ref")
    return doc["components"]["schemas"][ref.rsplit("/", 1)[-1]] if ref else schema


def has_example(doc, media):
    return bool(
        media.get("example")
        or media.get("examples")
        or resolve(doc, media["schema"]).get("examples")
    )


def test_toda_rota_tem_exemplos_de_requisicao_e_resposta():
    doc = spec()
    for path, ops in doc["paths"].items():
        for method, op in ops.items():
            if "requestBody" in op:
                media = op["requestBody"]["content"]["application/json"]
                assert has_example(doc, media), f"{method} {path}: corpo sem exemplo"
            for status, response in op["responses"].items():
                media = response.get("content", {}).get("application/json")
                assert media is None or has_example(doc, media), f"{method} {path} {status}"


def test_patch_de_preferencias_tem_exemplos_nomeados():
    op = spec()["paths"]["/users/{user_id}/preferences"]["patch"]
    examples = op["requestBody"]["content"]["application/json"]["examples"]

    assert set(examples) == {"adicionar", "nao_gostei", "remover"}
    for example in examples.values():
        PreferencesPatch.model_validate(example["value"])


@pytest.mark.parametrize(
    "schema",
    [
        GameCreate,
        GameOut,
        HealthOut,
        InteractionIn,
        InteractionOut,
        ModelInfoOut,
        PreferencesPatch,
        PreferencesReplace,
        RecommendationsOut,
        RetrainOut,
        SimilarGamesOut,
        UserCreate,
        UserDetail,
        UserOut,
    ],
)
def test_exemplos_dos_schemas_sao_validos(schema):
    examples = schema.model_json_schema()["examples"]

    assert examples
    for example in examples:
        schema.model_validate(example)


def test_exemplos_de_erro_usam_o_envelope_e_a_mensagem_da_rota():
    op = spec()["paths"]["/games/{game_id}"]["get"]
    example = op["responses"]["404"]["content"]["application/json"]["example"]

    assert ErrorResponse.model_validate(example).error.message == "Jogo 99999 não encontrado"
    for status in (404, 409, 422, 503):
        assert ErrorResponse.model_validate(error_example(status)).error.code
