"""Rota de verificação de saúde da API."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Status da API")
def health(request: Request) -> dict:
    """Indica se a API está no ar, se há um modelo em memória e a versão dele.

    O modelo é carregado no primeiro pedido de recomendação.
    """
    service = getattr(request.app.state, "recommender", None)
    return {
        "status": "ok",
        "version": request.app.version,
        "model_loaded": service is not None,
        "model_version": service.version if service else None,
    }
