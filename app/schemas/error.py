"""Formato padrão das respostas de erro da API."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Problema em um campo específico (erros de validação)."""

    field: str = Field(description="Caminho do campo, ex.: 'body.games.0.hours'.")
    message: str


class ErrorBody(BaseModel):
    code: str = Field(description="Código estável do erro, ex.: 'not_found'.")
    message: str = Field(description="Mensagem legível, em português.")
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """Envelope usado em todas as respostas de erro."""

    error: ErrorBody


_DESCRIPTIONS = {
    404: "Recurso não encontrado",
    409: "Conflito com dados existentes",
    422: "Dados inválidos na requisição",
    503: "Modelo indisponível para a estratégia pedida",
}


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Documentação OpenAPI das respostas de erro de uma rota."""
    return {
        code: {"model": ErrorResponse, "description": _DESCRIPTIONS.get(code, "Erro")}
        for code in status_codes
    }
