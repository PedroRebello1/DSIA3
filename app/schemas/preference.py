"""Schemas de atualização das preferências (histórico) do usuário."""

from typing import Self

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from app.schemas.interaction import MAX_HOURS, InteractionIn
from app.schemas.user import ensure_unique_games


class PreferencesReplace(BaseModel):
    """Conjunto completo de preferências; substitui o histórico atual (PUT)."""

    games: list[InteractionIn] = Field(description="Novo histórico completo do usuário.")

    _unique_games = field_validator("games")(ensure_unique_games)


class InteractionPatch(BaseModel):
    """Alteração de um jogo; campos omitidos mantêm o valor atual (ou o padrão, se for novo)."""

    game_id: int = Field(gt=0, description="ID do jogo.")
    purchased: bool | None = Field(None, description="Se o usuário comprou o jogo.")
    hours: float | None = Field(None, ge=0, le=MAX_HOURS, description="Horas jogadas.")
    disliked: bool | None = Field(None, description='Marca ou desmarca "não gostei".')


class PreferencesPatch(BaseModel):
    """Alterações parciais no histórico (PATCH)."""

    upsert: list[InteractionPatch] = Field(
        default_factory=list, description="Jogos a adicionar ou atualizar."
    )
    remove: list[PositiveInt] = Field(
        default_factory=list, description="IDs de jogos a remover do histórico."
    )

    _unique_games = field_validator("upsert")(ensure_unique_games)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if not self.upsert and not self.remove:
            raise ValueError("informe ao menos um jogo em 'upsert' ou 'remove'")
        overlap = {item.game_id for item in self.upsert} & set(self.remove)
        if overlap:
            raise ValueError(f"jogos em 'upsert' e 'remove' ao mesmo tempo: {sorted(overlap)}")
        return self
