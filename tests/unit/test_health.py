from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health_responde_ok():
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_settings_le_variaveis_de_ambiente(monkeypatch):
    monkeypatch.setenv("DEFAULT_K", "5")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    settings = Settings(_env_file=None)

    assert settings.default_k == 5
    assert settings.database_url == "sqlite://"
