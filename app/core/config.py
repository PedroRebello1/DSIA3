"""Configurações da aplicação, lidas de variáveis de ambiente ou do arquivo ``.env``."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros configuráveis da API."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Steam RecSys API"
    app_version: str = "0.1.0"
    database_url: str = "postgresql+psycopg://steam:steam@localhost:5432/steam"
    artifacts_dir: str = "artifacts"
    model_filename: str = "model.joblib"
    default_k: int = 10

    @property
    def model_path(self) -> Path:
        """Caminho do artefato do modelo treinado (gerado por ``recsys.train``)."""
        return Path(self.artifacts_dir) / self.model_filename


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instância única de ``Settings`` (cacheada)."""
    return Settings()
