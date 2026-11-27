"""Rota de verificação de saúde da API."""

from fastapi import APIRouter, Request

from app.schemas.health import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Status da API")
def health(request: Request) -> HealthOut:
    """Indica se a API está no ar, se há um modelo em memória e a versão dele.

    O modelo treinado é carregado de `artifacts/` na subida da API. Sem artefato,
    `model_loaded` fica `false` até o primeiro pedido de recomendação, que cria um modelo
    de popularidade a partir do banco.
    """
    service = getattr(request.app.state, "recommender", None)
    return HealthOut(
        status="ok",
        version=request.app.version,
        model_loaded=service is not None,
        model_version=service.version if service else None,
    )
