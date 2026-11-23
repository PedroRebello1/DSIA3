"""Rota de verificação de saúde da API."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Status da API")
def health(request: Request) -> dict:
    """Indica se a API está no ar e se há um modelo carregado em memória."""
    model = getattr(request.app.state, "model", None)
    return {
        "status": "ok",
        "version": request.app.version,
        "model_loaded": model is not None,
    }
