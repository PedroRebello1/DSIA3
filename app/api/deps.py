"""Dependências compartilhadas pelas rotas."""

from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db

DbSession = Annotated[Session, Depends(get_db)]
Limit = Annotated[int, Query(ge=1, le=100, description="Máximo de itens na resposta.")]
Offset = Annotated[int, Query(ge=0, description="Quantos itens pular (paginação).")]
