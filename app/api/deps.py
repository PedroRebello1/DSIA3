"""Dependências compartilhadas pelas rotas."""

from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.recommender_service import RecommenderService

DbSession = Annotated[Session, Depends(get_db)]
Limit = Annotated[int, Query(ge=1, le=100, description="Máximo de itens na resposta.")]
Offset = Annotated[int, Query(ge=0, description="Quantos itens pular (paginação).")]


def get_recommender_service(request: Request, db: DbSession) -> RecommenderService:
    """Serviço de recomendação da aplicação; o modelo é treinado no primeiro uso."""
    service = getattr(request.app.state, "recommender", None)
    if service is None:
        service = RecommenderService.from_db(db)
        request.app.state.recommender = service
    return service


Recommender = Annotated[RecommenderService, Depends(get_recommender_service)]
