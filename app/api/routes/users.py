"""Rotas de usuários."""

from fastapi import APIRouter, status

from app.api.deps import DbSession, Limit, Offset
from app.schemas.error import error_responses
from app.schemas.user import UserCreate, UserDetail, UserOut
from app.services import users as service

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "",
    response_model=UserDetail,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(404, 409, 422),
)
def create_user(data: UserCreate, db: DbSession) -> UserDetail:
    """Cria um usuário, opcionalmente já com jogos no histórico."""
    return service.create_user(db, data)


@router.get("", response_model=list[UserOut], responses=error_responses(422))
def list_users(db: DbSession, limit: Limit = 20, offset: Offset = 0) -> list[UserOut]:
    """Lista usuários por ordem de ID."""
    return service.list_users(db, limit, offset)


@router.get("/{user_id}", response_model=UserDetail, responses=error_responses(404, 422))
def get_user(user_id: int, db: DbSession) -> UserDetail:
    """Dados do usuário e seu histórico de jogos."""
    return service.get_user(db, user_id)
