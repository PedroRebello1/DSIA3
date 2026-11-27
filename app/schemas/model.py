"""Schemas de informações do modelo."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RUN_EXAMPLE = {
    "id": 1,
    "trained_at": "2026-11-27T14:00:00",
    "params": {
        "alpha": 0.25,
        "k_neighbors": 20,
        "weights": [[0, [0.0, 0.0, 1.0]], [1, [0.3, 0.3, 0.4]], [3, [0.7, 0.2, 0.1]]],
        "dislike_penalty": 0.5,
    },
    "metrics": {
        "precision@10": 0.1421,
        "recall@10": 0.4027,
        "ndcg@10": 0.3975,
        "map@10": 0.3186,
        "hit_rate@10": 0.6944,
        "coverage": 0.3846,
        "novelty": 6.2372,
        "users": 3757,
    },
    "artifact_path": "artifacts/model.joblib",
}


class ModelRunOut(BaseModel):
    """Um treino registrado em ``model_runs``."""

    model_config = ConfigDict(from_attributes=True, json_schema_extra={"examples": [RUN_EXAMPLE]})

    id: int = Field(description="ID do treino.")
    trained_at: datetime = Field(description="Data/hora do treino.")
    params: dict[str, Any] = Field(description="Hiperparâmetros usados.")
    metrics: dict[str, Any] = Field(description="Métricas da avaliação offline.")
    artifact_path: str = Field(description="Arquivo do modelo treinado.")


class ModelInfoOut(BaseModel):
    """Modelo em memória na API e último treino registrado."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "loaded": True,
                    "strategy": "hybrid",
                    "version": "2026-11-27T14:00:00+00:00",
                    "last_run": RUN_EXAMPLE,
                    "retraining": False,
                }
            ]
        }
    )

    loaded: bool = Field(description="Se há um modelo carregado em memória.")
    strategy: str | None = Field(description="Estratégia padrão do modelo carregado.")
    version: str | None = Field(description="Versão (data do treino) do modelo carregado.")
    last_run: ModelRunOut | None = Field(description="Último treino registrado no banco.")
    retraining: bool = Field(False, description="Se há um retreino em andamento.")


class RetrainOut(BaseModel):
    """Confirmação de que o retreino começou."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "started", "message": "Retreino iniciado; acompanhe em GET /model/info."}
            ]
        }
    )

    status: str = Field(description="Sempre 'started'.")
    message: str = Field(description="Como acompanhar o retreino.")
