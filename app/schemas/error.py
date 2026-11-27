"""Formato padrão das respostas de erro da API."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Problema em um campo específico (erros de validação)."""

    field: str = Field(description="Caminho do campo, ex.: 'body.games.0.hours'.")
    message: str


class ErrorBody(BaseModel):
    """Conteúdo do erro: código, mensagem e detalhes."""

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

# Código, mensagem e detalhes de exemplo de cada status (as rotas podem trocar a mensagem).
_EXAMPLES: dict[int, tuple[str, str, list[dict[str, str]] | None]] = {
    404: ("not_found", "Usuário 99999 não encontrado", None),
    409: ("conflict", "Já existe o jogo 'Fallout 4' (id 2)", None),
    422: (
        "validation_error",
        "Dados inválidos na requisição",
        [{"field": "body.games.0.hours", "message": "deve ser maior ou igual a 0"}],
    ),
    503: (
        "model_unavailable",
        "Estratégia 'cf' indisponível: o modelo híbrido não está carregado (disponíveis: "
        "popular). Treine o modelo com POST /model/retrain.",
        None,
    ),
}


def error_example(status_code: int, message: str | None = None) -> dict[str, Any]:
    """Corpo de exemplo de um erro, no formato do envelope padrão."""
    code, default_message, details = _EXAMPLES[status_code]
    return {"error": {"code": code, "message": message or default_message, "details": details}}


def error_responses(
    *status_codes: int, messages: dict[int, str] | None = None
) -> dict[int | str, dict[str, Any]]:
    """Documentação OpenAPI das respostas de erro de uma rota, com um exemplo de cada.

    Args:
        messages: mensagens de exemplo específicas da rota, por status.
    """
    messages = messages or {}
    return {
        code: {
            "model": ErrorResponse,
            "description": _DESCRIPTIONS.get(code, "Erro"),
            "content": {"application/json": {"example": error_example(code, messages.get(code))}},
        }
        for code in status_codes
    }
