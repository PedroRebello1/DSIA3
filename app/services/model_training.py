"""Retreino do modelo em segundo plano, com troca atômica do modelo em uso.

O treino roda fora da requisição (``BackgroundTasks``). Enquanto isso a API continua
respondendo com o modelo antigo; ao terminar, ``app.state.recommender`` passa a apontar
para o novo serviço em uma única atribuição. Só um retreino roda por vez.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import Engine

from app.core.errors import ConflictError
from app.services.recommender_service import RecommenderService
from recsys.train import load_from_db, register_run, train

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def is_retraining(app: FastAPI) -> bool:
    """Se há um retreino em andamento."""
    return bool(getattr(app.state, "retraining", False))


def start_retrain(app: FastAPI) -> None:
    """Marca o início de um retreino; 409 se já houver um em andamento."""
    with _lock:
        if is_retraining(app):
            raise ConflictError("Já existe um retreino em andamento")
        app.state.retraining = True


def retrain_model(app: FastAPI, engine: Engine, artifacts_dir: str | Path, search: bool) -> None:
    """Treina com os dados atuais do banco, registra o treino e troca o modelo em uso."""
    try:
        interactions, games = load_from_db(engine)
        result = train(interactions, games, artifacts_dir=artifacts_dir, search=search)
        register_run(engine, result)
        version = result.model.metadata_["trained_at"]
        new_service = RecommenderService(
            result.model, "hybrid", version, alpha=float(result.params["alpha"])
        )
        app.state.recommender = new_service  # troca atômica
        logger.info("Retreino concluído; modelo %s em uso", version)
    except Exception:
        logger.exception("Falha no retreino; o modelo anterior continua em uso")
    finally:
        app.state.retraining = False
