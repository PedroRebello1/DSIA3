"""Erros de domínio e conversão de todos os erros para o formato padrão da API.

Toda resposta de erro tem o formato::

    {"error": {"code": "not_found", "message": "Usuário 99 não encontrado", "details": null}}
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.error import ErrorBody, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    503: "model_unavailable",
}

# O Starlette e o Pydantic geram as mensagens em inglês; estas as substituem.
HTTP_MESSAGES = {
    404: "Rota não encontrada",
    405: "Método HTTP não permitido nesta rota",
}
VALIDATION_MESSAGES = {
    "missing": "campo obrigatório",
    "greater_than": "deve ser maior que {gt}",
    "greater_than_equal": "deve ser maior ou igual a {ge}",
    "less_than": "deve ser menor que {lt}",
    "less_than_equal": "deve ser menor ou igual a {le}",
    "string_too_short": "deve ter pelo menos {min_length} caractere(s)",
    "string_too_long": "deve ter no máximo {max_length} caracteres",
    "too_short": "deve ter pelo menos {min_length} item(ns)",
    "too_long": "deve ter no máximo {max_length} itens",
    "int_parsing": "deve ser um número inteiro",
    "int_type": "deve ser um número inteiro",
    "int_from_float": "deve ser um número inteiro",
    "float_parsing": "deve ser um número",
    "float_type": "deve ser um número",
    "bool_parsing": "deve ser true ou false",
    "bool_type": "deve ser true ou false",
    "string_type": "deve ser um texto",
    "list_type": "deve ser uma lista",
    "dict_type": "deve ser um objeto JSON",
    "model_attributes_type": "deve ser um objeto JSON",
    "json_invalid": "JSON inválido",
    "enum": "deve ser um destes valores: {expected}",
}


class AppError(Exception):
    """Erro de regra de negócio com o status HTTP correspondente."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    """Recurso (usuário, jogo) inexistente: 404."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(AppError):
    """Conflito com dados existentes: 409."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class ModelUnavailableError(AppError):
    """A estratégia pedida depende de um modelo que não está carregado."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_unavailable"


def error_response(
    status_code: int, code: str, message: str, details: list[ErrorDetail] | None = None
) -> JSONResponse:
    """Resposta JSON com o envelope padrão de erro."""
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Converte os erros de domínio (``AppError``) para HTTP."""
    return error_response(exc.status_code, exc.code, exc.message)


def _format_ctx(value: Any) -> str:
    """Valor de contexto legível: ``0.0`` vira ``0`` e "a or b" vira "a ou b"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(" or ", " ou ")


def validation_message(error: dict[str, Any]) -> str:
    """Mensagem em português de um erro do Pydantic (a original, se o tipo não for conhecido)."""
    if error["type"] == "value_error":  # mensagens dos nossos validadores, já em português
        return error["msg"].removeprefix("Value error, ")
    template = VALIDATION_MESSAGES.get(error["type"])
    if template is None:
        return error["msg"]
    ctx = {key: _format_ctx(value) for key, value in error.get("ctx", {}).items()}
    try:
        return template.format(**ctx)
    except KeyError:
        return error["msg"]


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 com o campo e o problema de cada erro de validação."""
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in err["loc"]), message=validation_message(err)
        )
        for err in exc.errors()
    ]
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "validation_error",
        "Dados inválidos na requisição",
        details,
    )


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Erros HTTP do framework (rota inexistente, método não permitido)."""
    code = HTTP_CODES.get(exc.status_code, "http_error")
    message = str(exc.detail)
    if message == HTTPStatus(exc.status_code).phrase:  # mensagem padrão do Starlette
        message = HTTP_MESSAGES.get(exc.status_code, message)
    response = error_response(exc.status_code, code, message)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    """Violação de restrição do banco não tratada pelos serviços: 409."""
    logger.warning("Violação de integridade não tratada: %s", exc.orig)
    return error_response(
        status.HTTP_409_CONFLICT, "conflict", "A operação conflita com dados existentes"
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 genérico; o detalhe vai só para o log."""
    logger.exception("Erro inesperado em %s %s", request.method, request.url.path)
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "Erro interno do servidor"
    )


def register_error_handlers(app: FastAPI) -> None:
    """Registra os conversores de erro na aplicação."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(IntegrityError, integrity_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
