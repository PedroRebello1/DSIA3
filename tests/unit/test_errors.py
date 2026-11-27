import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from app.core.errors import (
    ConflictError,
    NotFoundError,
    register_error_handlers,
    validation_message,
)
from app.schemas.error import error_responses


class Payload(BaseModel):
    n: int = Field(ge=0)


@pytest.fixture
def client():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/not-found")
    def not_found():
        raise NotFoundError("Usuário 9 não encontrado")

    @app.get("/conflict")
    def conflict():
        raise ConflictError("Já existe")

    @app.get("/integrity")
    def integrity():
        raise IntegrityError("INSERT ...", {}, Exception("unique"))

    @app.get("/boom")
    def boom():
        raise RuntimeError("falha inesperada")

    @app.post("/payload")
    def payload(data: Payload):
        return data

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("path", "status", "code", "message"),
    [
        ("/not-found", 404, "not_found", "Usuário 9 não encontrado"),
        ("/conflict", 409, "conflict", "Já existe"),
        ("/integrity", 409, "conflict", "A operação conflita com dados existentes"),
        ("/boom", 500, "internal_error", "Erro interno do servidor"),
        ("/rota-inexistente", 404, "not_found", "Rota não encontrada"),
    ],
)
def test_erros_no_formato_padrao(client, path, status, code, message):
    response = client.get(path)

    assert response.status_code == status
    assert response.json() == {"error": {"code": code, "message": message, "details": None}}


def test_erro_de_validacao_lista_os_campos(client):
    response = client.post("/payload", json={"n": -1})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == "body.n"
    assert error["details"][0]["message"] == "deve ser maior ou igual a 0"


def test_metodo_nao_permitido(client):
    response = client.delete("/conflict")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert response.json()["error"]["message"] == "Método HTTP não permitido nesta rota"
    assert "GET" in response.headers["allow"]


def test_error_responses_documenta_o_envelope():
    docs = error_responses(404, 422)

    assert set(docs) == {404, 422}
    assert docs[404]["description"] == "Recurso não encontrado"
    assert docs[422]["model"].__name__ == "ErrorResponse"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        ({"type": "missing", "msg": "Field required"}, "campo obrigatório"),
        (
            {"type": "greater_than", "msg": "Input should be greater than 0", "ctx": {"gt": 0}},
            "deve ser maior que 0",
        ),
        (
            {"type": "string_too_long", "msg": "...", "ctx": {"max_length": 255}},
            "deve ter no máximo 255 caracteres",
        ),
        (
            {"type": "enum", "msg": "...", "ctx": {"expected": "'hybrid', 'cf' or 'popular'"}},
            "deve ser um destes valores: 'hybrid', 'cf' ou 'popular'",
        ),
        (
            {"type": "less_than_equal", "msg": "...", "ctx": {"le": 100000.0}},
            "deve ser menor ou igual a 100000",
        ),
        ({"type": "value_error", "msg": "Value error, jogos repetidos"}, "jogos repetidos"),
        ({"type": "tipo_desconhecido", "msg": "Original message"}, "Original message"),
        ({"type": "greater_than", "msg": "Sem contexto"}, "Sem contexto"),
    ],
)
def test_mensagens_de_validacao_em_portugues(error, message):
    assert validation_message(error) == message


def test_validacao_da_api_responde_em_portugues(client):
    response = client.post("/payload", json={"n": "abc"})

    assert response.json()["error"]["details"] == [
        {"field": "body.n", "message": "deve ser um número inteiro"}
    ]


def test_corpo_que_nao_e_utf8_tem_mensagem_em_portugues(client):
    body = '{"n": 1, "nome": "ação"}'.encode("latin-1")
    response = client.post("/payload", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "bad_request",
        "message": "Não foi possível ler o corpo da requisição; envie JSON válido em UTF-8",
        "details": None,
    }
