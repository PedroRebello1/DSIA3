"""Schemas de informações do modelo."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelRunOut(BaseModel):
    """Um treino registrado em ``model_runs``."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="ID do treino.")
    trained_at: datetime = Field(description="Data/hora do treino.")
    params: dict[str, Any] = Field(description="Hiperparâmetros usados.")
    metrics: dict[str, Any] = Field(description="Métricas da avaliação offline.")
    artifact_path: str = Field(description="Arquivo do modelo treinado.")


class ModelInfoOut(BaseModel):
    """Modelo em memória na API e último treino registrado."""

    loaded: bool = Field(description="Se há um modelo carregado em memória.")
    strategy: str | None = Field(description="Estratégia padrão do modelo carregado.")
    version: str | None = Field(description="Versão (data do treino) do modelo carregado.")
    last_run: ModelRunOut | None = Field(description="Último treino registrado no banco.")
    retraining: bool = Field(False, description="Se há um retreino em andamento.")


class RetrainOut(BaseModel):
    """Confirmação de que o retreino começou."""

    status: str = Field(description="Sempre 'started'.")
    message: str = Field(description="Como acompanhar o retreino.")
