import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from app.core.errors import ConflictError, NotFoundError, register_error_handlers
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
        ("/rota-inexistente", 404, "not_found", "Not Found"),
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
    assert "greater than or equal to 0" in error["details"][0]["message"]


def test_metodo_nao_permitido(client):
    response = client.delete("/conflict")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in response.headers["allow"]


def test_error_responses_documenta_o_envelope():
    docs = error_responses(404, 422)

    assert set(docs) == {404, 422}
    assert docs[404]["description"] == "Recurso não encontrado"
    assert docs[422]["model"].__name__ == "ErrorResponse"
