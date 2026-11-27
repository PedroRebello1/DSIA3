"""Schema da verificação de saúde."""

from pydantic import BaseModel, ConfigDict, Field


class HealthOut(BaseModel):
    """Status da API e do modelo em memória."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "version": "0.1.0",
                    "model_loaded": True,
                    "model_version": "2026-11-27T14:00:00+00:00",
                }
            ]
        }
    )

    status: str = Field(description="Sempre 'ok' quando a API responde.")
    version: str = Field(description="Versão da API.")
    model_loaded: bool = Field(description="Se há um modelo de recomendação em memória.")
    model_version: str | None = Field(description="Versão (data do treino) do modelo carregado.")
