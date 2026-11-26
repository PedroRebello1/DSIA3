"""Informações sobre o modelo carregado e os treinos registrados."""

from sqlalchemy.orm import Session

from app.db import crud
from app.schemas.model import ModelInfoOut, ModelRunOut
from app.services.recommender_service import RecommenderService


def get_model_info(
    db: Session, service: RecommenderService | None, retraining: bool = False
) -> ModelInfoOut:
    run = crud.get_latest_model_run(db)
    return ModelInfoOut(
        loaded=service is not None,
        strategy=service.strategy if service else None,
        version=service.version if service else None,
        last_run=ModelRunOut.model_validate(run) if run else None,
        retraining=retraining,
    )
